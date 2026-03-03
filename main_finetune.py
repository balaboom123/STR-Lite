import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from engine_finetune import evaluate, evaluate_per_benchmark, print_benchmark_results, train_one_epoch
from src.data.lmdb_dataset import build_lmdb_dataset, build_lmdb_datasets_by_name, lmdb_collate_fn
from src.models import vit_str_ar as vit_models
from src.tokenizer import CharsetTokenizer
from util.lr_decay import param_groups_lrd
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def _cfg_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("hydra", None)
    return SimpleNamespace(**cfg_dict)


def load_pretrained_mae_encoder(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    load_state = {}

    for key, value in state_dict.items():
        target_key = key[7:] if key.startswith("module.") else key

        if target_key.startswith("decoder_") or target_key.startswith("mask_token"):
            continue

        if target_key not in model_state:
            target_key = f"encoder.{target_key}"

        if target_key in model_state and model_state[target_key].shape == value.shape:
            load_state[target_key] = value

    if len(load_state) == 0:
        raise RuntimeError(
            f"No encoder parameters loaded from {ckpt_path}. "
            "Check that the checkpoint contains MAE encoder weights."
        )

    msg = model.load_state_dict(load_state, strict=False)
    print(f"Loaded MAE encoder from {ckpt_path}")
    print(f"Loaded params: {len(load_state)}")
    print(msg)


def run(args: SimpleNamespace, cfg: DictConfig):
    if getattr(args, "eval_max_decode_len", None) is None:
        args.eval_max_decode_len = args.max_label_length

    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    tokenizer = CharsetTokenizer(
        character_dict_path=args.character_dict_path,
        max_text_length=args.max_label_length,
        lower=args.lower,
        use_space_char=args.use_space_char,
    )

    dataset_train = build_lmdb_dataset(
        root_dir=args.train_data_path,
        img_height=args.img_height,
        img_width=args.img_width,
        return_label=True,
        max_label_length=args.max_label_length,
        label_encoder=tokenizer.label_encoder,
        augment=True,
        randaugment_layers=args.randaugment_layers,
        randaugment_magnitude=args.randaugment_magnitude,
        randaugment_prob=args.randaugment_prob,
        readahead=args.readahead,
    )
    dataset_val = build_lmdb_dataset(
        root_dir=args.val_data_path,
        img_height=args.img_height,
        img_width=args.img_width,
        return_label=True,
        max_label_length=args.max_label_length,
        label_encoder=tokenizer.label_encoder,
        augment=False,
        randaugment_layers=args.randaugment_layers,
        randaugment_magnitude=args.randaugment_magnitude,
        randaugment_prob=0.0,
        readahead=args.readahead,
    )
    print(dataset_train)
    print(dataset_val)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not getattr(args, "eval", False):
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=lmdb_collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=lmdb_collate_fn,
    )

    if not hasattr(vit_models, args.model):
        raise ValueError(f"Unknown model: {args.model}")
    model = getattr(vit_models, args.model)(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        img_size=(args.img_height, args.img_width),
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        dropout=args.decoder_dropout,
    )

    if args.pretrained_mae and not args.resume:
        load_pretrained_mae_encoder(model, args.pretrained_mae)

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model = {model_without_ddp}")
    print(f"number of params (M): {n_parameters / 1e6:.2f}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.no_layer_decay:
        param_groups = [
            {
                "params": [p for _, p in model_without_ddp.named_parameters() if p.requires_grad and p.ndim > 1],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for _, p in model_without_ddp.named_parameters() if p.requires_grad and p.ndim <= 1],
                "weight_decay": 0.0,
            },
        ]
    else:
        param_groups = param_groups_lrd(
            model_without_ddp,
            weight_decay=args.weight_decay,
            layer_decay=args.layer_decay,
            no_weight_decay_list=(),
        )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(optimizer)

    loss_scaler = NativeScaler(enabled=str(args.precision).lower() == "fp16")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    if args.resume:
        args.start_epoch = misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    if getattr(args, "eval", False):
        eval_paths = args.val_data_path
        benchmark_datasets = build_lmdb_datasets_by_name(
            root_dir=eval_paths,
            img_height=args.img_height,
            img_width=args.img_width,
            return_label=True,
            max_label_length=args.max_label_length,
            label_encoder=tokenizer.label_encoder,
            augment=False,
            readahead=args.readahead,
        )
        results = evaluate_per_benchmark(
            benchmark_datasets=benchmark_datasets,
            model=model,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            precision=args.precision,
            max_decode_len=args.eval_max_decode_len,
            distributed=getattr(args, "distributed", False),
        )
        print_benchmark_results(results)
        return

    print(f"Start fine-tuning for {args.epochs} epochs")
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            tokenizer=tokenizer,
            log_writer=log_writer,
            args=args,
        )

        test_stats = evaluate(
            data_loader_val,
            model,
            criterion,
            tokenizer,
            device,
            precision=args.precision,
            max_decode_len=args.eval_max_decode_len,
        )
        epoch_acc = 100.0 * test_stats["acc"]
        print(f"Word accuracy on {len(dataset_val)} val images: {epoch_acc:.3f}%")

        is_best = test_stats["acc"] >= best_acc
        best_acc = max(best_acc, test_stats["acc"])
        print(f"Max word accuracy: {100.0 * best_acc:.3f}%")

        if args.output_dir:
            if args.save_best_only:
                if is_best:
                    misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        tag="best",
                    )
            else:
                if (epoch % args.save_freq == 0) or (epoch + 1 == args.epochs):
                    misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                    )

        if log_writer is not None:
            log_writer.add_scalar("perf/acc", test_stats["acc"], epoch)
            log_writer.add_scalar("perf/acc_real", test_stats["acc_real"], epoch)
            log_writer.add_scalar("perf/acc_lower", test_stats["acc_lower"], epoch)
            log_writer.add_scalar("perf/val_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
            "best_acc": best_acc,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


@hydra.main(config_path="conf", config_name="finetune", version_base=None)
def main(cfg: DictConfig):
    args = _cfg_to_namespace(cfg)

    # Use Hydra's runtime output directory for all outputs (timestamped)
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    args.output_dir = hydra_output_dir
    args.log_dir = hydra_output_dir

    if isinstance(args.train_data_path, str):
        args.train_data_path = [args.train_data_path]
    if isinstance(args.val_data_path, str):
        args.val_data_path = [args.val_data_path]
    if hasattr(args, "test_data_path") and isinstance(args.test_data_path, str):
        args.test_data_path = [args.test_data_path]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run(args, cfg)


if __name__ == "__main__":
    main()
