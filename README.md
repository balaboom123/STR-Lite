# MAE-for-Efficient-Scene-Text-Recognition

Tiny MAE pretraining + autoregressive Transformer-decoder fine-tuning for scene text recognition using **normal LMDB**, managed by **Hydra** configs.

## Core Setup

- Input: `32 x 128` RGB
- Patch: `4 x 8` -> `8 x 16 = 128` tokens
- Tiny encoder: `depth=12`, `embed_dim=192`, `heads=12`
- MAE decoder (pretrain): `depth=1`, `embed_dim=96`, `heads=3`
- Positional embedding (pretrain/fine-tune encoder): fixed 2D sin-cos (original MAE style)
- Fine-tune head: ViT encoder -> TransformerDecoder (autoregressive) -> token logits
- Inference: greedy autoregressive decoding with per-layer cache
- Character dict: `util/EN_symbol_dict.txt`
- Fixed runtime settings: `max_label_length=25`, `seed=0`, `device=cuda`, `precision=bf16`
- Data transfer optimization: CPU batches use `uint8` images and `int16` text ids, then cast/normalize on GPU for training.

## Hydra Configs

- Pretrain config: `conf/pretrain.yaml`
- Fine-tune config: `conf/finetune.yaml`

## LMDB Format

Each LMDB must contain:

- `num-samples`
- `image-000000001`, `image-000000002`, ...
- `label-000000001`, `label-000000002`, ... (required for fine-tuning)

`data_path`, `train_data_path`, and `val_data_path` support one or many roots. Any nested folder containing `data.mdb` is discovered recursively.

## Install

```bash
pip install -r requirements.txt
```

## Pretraining (Hydra)

```bash
python main_pretrain.py data_path='[/path/to/lmdb_pretrain]'
```

Distributed example:

```bash
torchrun --nproc_per_node=8 main_pretrain.py \
  data_path='[/path/to/lmdb_pretrain]' \
  output_dir=./output/pretrain_tiny \
  log_dir=./output/pretrain_tiny
```

## Fine-tuning (Autoregressive)

```bash
python main_finetune.py \
  train_data_path='[/path/to/lmdb_train]' \
  val_data_path='[/path/to/lmdb_val]' \
  pretrained_mae=./output/pretrain_tiny/checkpoint-last.pth
```

Evaluate:

```bash
python main_finetune.py \
  train_data_path='[/path/to/lmdb_train]' \
  val_data_path='[/path/to/lmdb_val]' \
  resume=./output/finetune/checkpoint-last.pth \
  eval=true
```

## Key Files

- `main_pretrain.py`: Hydra entrypoint for MAE pretraining
- `main_finetune.py`: Hydra entrypoint for autoregressive fine-tuning
- `conf/pretrain.yaml`: pretraining config
- `conf/finetune.yaml`: fine-tuning config
- `src/models/mae_vit_tiny_str.py`: tiny MAE
- `src/models/vit_str_ar.py`: ViT-tiny + TransformerDecoder (AR)
- `src/tokenizer.py`: BOS/EOS/PAD tokenizer with dict-backed charset
- `src/metrics/rec_metric.py`: eval metrics (`acc`, `acc_real`, `acc_lower`)
