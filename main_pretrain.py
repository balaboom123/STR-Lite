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
from engine_pretrain import train_one_epoch
from src.data.lmdb_dataset import build_lmdb_dataset, lmdb_collate_fn
from src.models import mae_vit_tiny_str as mae_models
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def _cfg_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("hydra", None)
    return SimpleNamespace(**cfg_dict)


def run(args: SimpleNamespace, cfg: DictConfig):
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_lmdb_dataset(
        root_dir=args.data_path,
        img_height=args.img_height,
        img_width=args.img_width,
        return_label=False,
        max_label_length=args.max_label_length,
        augment=True,
        randaugment_layers=args.randaugment_layers,
        randaugment_magnitude=args.randaugment_magnitude,
        randaugment_prob=args.randaugment_prob,
        readahead=args.readahead,
    )
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print(f"Sampler_train = {sampler_train}")

    if global_rank == 0 and args.log_dir is not None:
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

    if not hasattr(mae_models, args.model):
        raise ValueError(f"Unknown model: {args.model}")
    model = getattr(mae_models, args.model)(
        norm_pix_loss=args.norm_pix_loss,
        img_size=(args.img_height, args.img_width),
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
    )
    model.to(device)

    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

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
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler(enabled=str(args.precision).lower() == "fp16")

    if args.resume:
        args.start_epoch = misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and ((epoch % args.save_freq == 0) or (epoch + 1 == args.epochs)):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


@hydra.main(config_path="conf", config_name="pretrain", version_base=None)
def main(cfg: DictConfig):
    args = _cfg_to_namespace(cfg)

    # Use Hydra's runtime output directory for all outputs (timestamped)
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    args.output_dir = hydra_output_dir
    args.log_dir = hydra_output_dir

    if isinstance(args.data_path, str):
        args.data_path = [args.data_path]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run(args, cfg)


if __name__ == "__main__":
    main()
