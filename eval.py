"""Standalone per-benchmark evaluation for fine-tuned STR models."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

import util.misc as misc
from engine_finetune import evaluate_per_benchmark, print_benchmark_results
from src.data.lmdb_dataset import build_lmdb_datasets_by_name
from src.models import vit_str_ar as vit_models
from src.tokenizer import CharsetTokenizer


def _resolve_path(path, base_dir):
    """Resolve a path relative to base_dir, or return None if empty."""
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _resolve_paths(paths, base_dir):
    """Resolve a list of paths relative to base_dir."""
    if paths is None:
        return None
    if isinstance(paths, str):
        paths = [paths]
    resolved = []
    for p in paths:
        r = _resolve_path(p, base_dir)
        if r is not None:
            resolved.append(r)
    return resolved if resolved else None


def _prepare_eval_cfg(cfg, base_dir):
    """Merge base finetune config, config_resume, and CLI overrides."""
    cfg_overrides = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    config_resume = _resolve_path(cfg_overrides.get("config_resume"), base_dir)

    if config_resume and not os.path.isfile(config_resume):
        raise FileNotFoundError(f"config_resume not found: {config_resume}")

    # Remove empty override keys so they don't clobber merged values
    for key in ("resume", "test_data_path", "config_resume"):
        val = cfg_overrides.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            if key in cfg_overrides:
                del cfg_overrides[key]
        elif isinstance(val, (list, tuple)) and all(
            not str(v).strip() or str(v).strip() == "/path/to/data_lmdb_release/test"
            for v in val
        ):
            # Remove placeholder default test_data_path
            if key in cfg_overrides:
                del cfg_overrides[key]

    # Start with base finetune config
    base_cfg_path = os.path.join(base_dir, "conf", "finetune.yaml")
    if os.path.isfile(base_cfg_path):
        merged_cfg = OmegaConf.load(base_cfg_path)
    else:
        merged_cfg = OmegaConf.create({})

    # Merge saved training config on top (has actual model/data settings)
    if config_resume:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(config_resume))

    # Merge CLI overrides on top
    merged_cfg = OmegaConf.merge(merged_cfg, cfg_overrides)
    OmegaConf.set_struct(merged_cfg, False)
    merged_cfg.config_resume = config_resume or ""

    return merged_cfg


def run(args: SimpleNamespace):
    if getattr(args, "eval_max_decode_len", None) is None:
        args.eval_max_decode_len = args.max_label_length

    misc.init_distributed_mode(args)

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Tokenizer
    tokenizer = CharsetTokenizer(
        character_dict_path=args.character_dict_path,
        max_text_length=args.max_label_length,
        lower=args.lower,
        use_space_char=args.use_space_char,
    )

    # Model
    if not hasattr(vit_models, args.model):
        raise ValueError(f"Unknown model: {args.model}")
    model = getattr(vit_models, args.model)(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        img_size=(args.img_height, args.img_width),
        patch_size=args.patch_size,
        drop_path_rate=0.0,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        dropout=args.decoder_dropout,
    )
    model.to(device)

    # Load checkpoint
    if not args.resume:
        raise ValueError("resume (checkpoint path) is required for evaluation")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    # Strip "module." prefix if present (from DDP checkpoints)
    stripped = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    msg = model.load_state_dict(stripped, strict=False)
    print(f"Loaded checkpoint from {args.resume}")
    print(msg)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1e6:.2f}M")

    if getattr(args, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Build per-benchmark datasets from test_data_path
    if not args.test_data_path:
        raise ValueError("test_data_path must be set for evaluation.")

    benchmark_datasets = build_lmdb_datasets_by_name(
        root_dir=args.test_data_path,
        img_height=args.img_height,
        img_width=args.img_width,
        return_label=True,
        max_label_length=args.max_label_length,
        label_encoder=tokenizer.label_encoder,
        augment=False,
        readahead=args.readahead,
    )

    print(f"Found {len(benchmark_datasets)} benchmark(s): {list(benchmark_datasets.keys())}")

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

    # Save results to JSON
    if args.output_dir and misc.is_main_process():
        result_path = os.path.join(args.output_dir, "eval_results.json")
        payload = {
            "checkpoint": args.resume,
            "total": results["total"],
            "per_benchmark": {
                name: stats for name, stats in results["per_benchmark"].items()
            },
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved evaluation results to: {result_path}")


@hydra.main(config_path="conf", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    base_dir = hydra.utils.get_original_cwd()
    cfg = _prepare_eval_cfg(cfg, base_dir)

    # Resolve paths relative to project root
    cfg.resume = _resolve_path(cfg.get("resume"), base_dir)
    cfg.test_data_path = _resolve_paths(cfg.get("test_data_path"), base_dir)
    cfg.output_dir = _resolve_path(cfg.get("output_dir"), base_dir)

    # Convert to namespace
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("hydra", None)
    cfg_dict.pop("defaults", None)
    args = SimpleNamespace(**cfg_dict)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    run(args)


if __name__ == "__main__":
    main()
