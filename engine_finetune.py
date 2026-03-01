import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc
from src.metrics.rec_metric import RecMetric


class _DistributedEvalSampler(torch.utils.data.Sampler):
    """Shard eval data across ranks without padding or dropping samples."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        if num_replicas is None:
            num_replicas = misc.get_world_size()
        if rank is None:
            rank = misc.get_rank()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def _get_autocast_kwargs(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return {"enabled": False}
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    raise ValueError(f"Unsupported precision: {precision}")


def _prepare_images_for_model(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    # CPU uint8 -> GPU float tensor normalized to [-1, 1] for model input.
    images = images.to(device, non_blocking=True)
    images = images.to(dtype=torch.float32).div_(255.0)
    images = images.sub_(0.5).div_(0.5)
    return images


def _seq_ce_loss(logits: torch.Tensor, target_ids: torch.Tensor, criterion) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))


def _build_decoder_io(labels, tokenizer, device):
    if not isinstance(labels, dict):
        raise ValueError("Autoregressive finetune expects dict labels from lmdb_collate_fn")
    return tokenizer.build_decoder_inputs_from_text_ids(labels["text"], labels["lengths"], device=device)


def _get_model_unwrapped(model):
    """Unwrap DDP to access encode/decode methods."""
    return model.module if hasattr(model, "module") else model

def _compute_loss(model, criterion, images, labels, tokenizer, device, autocast_kwargs, mixup_alpha=0.0):
    images = _prepare_images_for_model(images, device)

    with torch.amp.autocast("cuda", **autocast_kwargs):
        if mixup_alpha > 0.0 and images.size(0) > 1:
            perm = torch.randperm(images.size(0), device=device)
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            mixed_images = lam * images + (1.0 - lam) * images[perm]

            labels_perm = {
                "text": labels["text"][perm.detach().cpu()],
                "lengths": labels["lengths"][perm.detach().cpu()],
            }

            in_a, tgt_a, pad_a = _build_decoder_io(labels, tokenizer, device)
            in_b, tgt_b, pad_b = _build_decoder_io(labels_perm, tokenizer, device)

            # Encode once, decode twice
            unwrapped = _get_model_unwrapped(model)
            memory = unwrapped.encode(mixed_images)
            logits_a = unwrapped.decode(memory, tgt_input=in_a, tgt_key_padding_mask=pad_a)
            logits_b = unwrapped.decode(memory, tgt_input=in_b, tgt_key_padding_mask=pad_b)

            loss_a = _seq_ce_loss(logits_a, tgt_a, criterion)
            loss_b = _seq_ce_loss(logits_b, tgt_b, criterion)
            loss = lam * loss_a + (1.0 - lam) * loss_b

            logits = logits_a
        else:
            in_ids, tgt_ids, pad_mask = _build_decoder_io(labels, tokenizer, device)
            logits = model(images, tgt_input=in_ids, tgt_key_padding_mask=pad_mask)
            loss = _seq_ce_loss(logits, tgt_ids, criterion)

    return loss, logits


def train_one_epoch(
    model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    tokenizer,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    # Log exactly first and last iteration in each epoch.
    print_freq = max(len(data_loader), 1)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    autocast_kwargs = _get_autocast_kwargs(device, getattr(args, "precision", "fp16"))

    for data_iter_step, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        loss, _ = _compute_loss(
            model=model,
            criterion=criterion,
            images=images,
            labels=labels,
            tokenizer=tokenizer,
            device=device,
            autocast_kwargs=autocast_kwargs,
            mixup_alpha=float(getattr(args, "mixup", 0.0)),
        )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion, tokenizer, device, precision="fp16", max_decode_len=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    autocast_kwargs = _get_autocast_kwargs(device, precision)
    rec_metric = RecMetric(valid_chars=tokenizer.character, is_lower=tokenizer.lower)

    # Pre-register the loss meter so that every rank participates in collective
    # sync even when the data loader is empty (e.g. rank >= dataset size).
    metric_logger.update(loss=0.0)
    metric_logger.meters["loss"].count = 0
    metric_logger.meters["loss"].total = 0.0

    for images, labels in metric_logger.log_every(data_loader, 20, header):
        with torch.amp.autocast("cuda", **autocast_kwargs):
            images_norm = _prepare_images_for_model(images, device)
            in_ids, tgt_ids, pad_mask = _build_decoder_io(labels, tokenizer, device)
            logits = model(images_norm, tgt_input=in_ids, tgt_key_padding_mask=pad_mask)
            loss = _seq_ce_loss(logits, tgt_ids, criterion)

        metric_logger.update(loss=loss.item())

        decoded_ids = model.greedy_decode(
            images_norm,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_decode_len,
        )
        pred_texts = tokenizer.decode_token_ids_batch(decoded_ids)
        gt_texts = tokenizer.normalize_batch(
            tokenizer.decode_char_ids_batch(labels["text"], labels["lengths"])
        )
        rec_metric.update_many(pred_texts, gt_texts)

    metric_logger.synchronize_between_processes()

    counts = rec_metric.get_counts()
    if misc.is_dist_avail_and_initialized():
        stats = torch.tensor(
            [
                counts["total"],
                counts["correct_num_real"],
                counts["correct_num_lower"],
                counts["correct_num"],
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(stats)
        counts = {
            "total": int(stats[0].item()),
            "correct_num_real": int(stats[1].item()),
            "correct_num_lower": int(stats[2].item()),
            "correct_num": int(stats[3].item()),
        }

    total = max(counts["total"], 1)
    acc = counts["correct_num"] / total
    acc_real = counts["correct_num_real"] / total
    acc_lower = counts["correct_num_lower"] / total

    print(
        f"* Acc {acc * 100.0:.3f} | Acc_real {acc_real * 100.0:.3f} | "
        f"Acc_lower {acc_lower * 100.0:.3f} | loss {metric_logger.loss.global_avg:.3f}"
    )

    return {
        "loss": metric_logger.loss.global_avg,
        "acc": acc,
        "acc_real": acc_real,
        "acc_lower": acc_lower,
        "total": counts["total"],
        "correct_num": counts["correct_num"],
        "correct_num_real": counts["correct_num_real"],
        "correct_num_lower": counts["correct_num_lower"],
    }


def evaluate_per_benchmark(
    benchmark_datasets,
    model,
    criterion,
    tokenizer,
    device,
    batch_size=256,
    num_workers=8,
    pin_memory=True,
    precision="fp16",
    max_decode_len=None,
    distributed=False,
):
    """Evaluate each benchmark dataset independently and aggregate totals.

    Args:
        benchmark_datasets: OrderedDict[name -> Dataset] from build_lmdb_datasets_by_name.

    Returns:
        dict with "per_benchmark" (OrderedDict[name -> stats]) and "total" (aggregated stats).
    """
    from collections import OrderedDict

    from src.data.lmdb_dataset import lmdb_collate_fn

    per_benchmark = OrderedDict()
    agg_total = 0
    agg_correct = 0
    agg_correct_real = 0
    agg_correct_lower = 0

    for name, dataset in benchmark_datasets.items():
        if distributed:
            sampler = _DistributedEvalSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=lmdb_collate_fn,
        )

        stats = evaluate(
            loader, model, criterion, tokenizer, device,
            precision=precision, max_decode_len=max_decode_len,
        )
        per_benchmark[name] = stats

        agg_total += stats["total"]
        agg_correct += stats["correct_num"]
        agg_correct_real += stats["correct_num_real"]
        agg_correct_lower += stats["correct_num_lower"]

    total = max(agg_total, 1)
    total_stats = {
        "acc": agg_correct / total,
        "acc_real": agg_correct_real / total,
        "acc_lower": agg_correct_lower / total,
        "total": agg_total,
        "correct_num": agg_correct,
        "correct_num_real": agg_correct_real,
        "correct_num_lower": agg_correct_lower,
    }

    return {"per_benchmark": per_benchmark, "total": total_stats}


def print_benchmark_results(results):
    """Pretty-print per-benchmark and total evaluation results."""
    print("\n" + "=" * 70)
    print(f"{'Benchmark':<25} {'Acc':>8} {'Acc_real':>10} {'Acc_lower':>10} {'Total':>8}")
    print("-" * 70)
    for name, stats in results["per_benchmark"].items():
        print(
            f"{name:<25} {stats['acc']*100:>7.2f}% "
            f"{stats['acc_real']*100:>9.2f}% "
            f"{stats['acc_lower']*100:>9.2f}% "
            f"{stats['total']:>8d}"
        )
    print("-" * 70)
    t = results["total"]
    print(
        f"{'TOTAL':<25} {t['acc']*100:>7.2f}% "
        f"{t['acc_real']*100:>9.2f}% "
        f"{t['acc_lower']*100:>9.2f}% "
        f"{t['total']:>8d}"
    )
    print("=" * 70 + "\n")
