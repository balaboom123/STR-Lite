import math
import sys
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc


def _get_autocast_kwargs(device: torch.device, args):
    precision = str(getattr(args, "precision", "fp16")).lower()
    if device.type != "cuda" or precision == "fp32":
        return {"enabled": False}
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    raise ValueError(f"Unsupported precision: {precision}")


def _prepare_images_for_model(samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    # CPU uint8 -> GPU float tensor normalized to [-1, 1] for model input.
    samples = samples.to(device, non_blocking=True)
    samples = samples.to(dtype=torch.float32).div_(255.0)
    samples = samples.sub_(0.5).div_(0.5)
    return samples


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
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

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    autocast_kwargs = _get_autocast_kwargs(device, args)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = _prepare_images_for_model(samples, device)

        with torch.cuda.amp.autocast(**autocast_kwargs):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

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
