<!-- H1 -->
# STRLite: MAE-Pretrained Scene Text Recognition

<!-- Animated Header -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=STRLite&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=MAE%20pretraining%20ViT%20in%20Lightweight&descAlignY=52&descSize=18" alt="STRLite banner"/>

STRLite trains scene text recognition models in two stages: MAE pretraining for visual representation learning, followed by autoregressive decoder fine-tuning for text generation.

Repository: https://github.com/balaboom123/STRLite

<div align="center">
  <img src="assets/architecture-overview.svg" width="800" />
</div>

</div>

## 1. Usage

### Installation Guide
We provide installation instructions in [INSTALLATION.md](INSTALLATION.md).

### Data Preparation
We describe how to prepare the datasets in [DATASET.md](DATASET.md).

## 2. STRLite

### 2.1. Pre-training
- ViT-Tiny pretrained on U14M-U.

  | Variants | Embedding | Depth | Heads | Parameters | Download |
  | -------- | :-------: | :---: | :---: | :--------: | :------: |
  | ViT-Tiny | 192 | 12 | 12 | 6M | [HuggingFace](https://huggingface.co/balaboom123/STRLite/resolve/main/pretrain/checkpoint-last.pth) |

- To pre-train the ViT backbone on your own dataset, see [§3.1 MAE Pretraining](#31-mae-pretraining).

### 2.2. Fine-tuning
- STRLite fine-tuned on U14M-L-Filtered.

  | Variants | Acc on Common Benchmarks | Acc on U14M-Benchmarks | Download |
  | -------- | :----------------------: | :--------------------: | :------: |
  | STRLite | 93.82 | 81.03 | [HuggingFace](https://huggingface.co/balaboom123/STRLite/resolve/main/finetune/checkpoint-best.pth) |

- To fine-tune or evaluate the model, see [§3.2 Fine-tuning](#32-fine-tuning) and [§3.3 Evaluation](#33-evaluation).

### 2.3 Results

Results of STRLite Accuracy (%) with or without MAE pretraining on six common Datasets.

<table>
<tr>
<td valign="top">

**Common STR benchmarks**

| Subset | w/ pretrain | w/o pretrain |
| ------ | :---------: | :----------: |
| CUTE80 | 95.83 | 94.79 |
| IC13 | 96.85 | 96.50 |
| IC15 | 86.80 | 86.25 |
| IIIT5k | 96.97 | 96.47 |
| SVT | 95.36 | 94.90 |
| SVTP | 92.40 | 89.77 |
| **Weighted avg.** | **93.82** | **93.12** |

</td>
<td valign="top">

**U14M benchmarks**

| Subset | w/ pretrain | w/o pretrain |
| --------------- | :---------: | :----------: |
| artistic | 67.78 | 62.11 |
| contextless | 78.95 | 77.43 |
| curve | 82.19 | 78.97 |
| general | 81.07 | 79.96 |
| multi oriented | 82.91 | 78.57 |
| multi words | 76.72 | 74.31 |
| salient | 78.17 | 75.33 |
| **Weighted avg.** | **81.03** | **79.88** |

</td>
</tr>
</table>

## 3. Quick Start

The end-to-end workflow is: pretrain a MAE encoder, fine-tune with an autoregressive decoder, then evaluate a checkpoint on validation or test benchmarks.

### 3.1 MAE Pretraining

```bash
python main_pretrain.py data_path='[/path/to/lmdb_pretrain]'
```

Distributed example:

```bash
torchrun --nproc_per_node=8 main_pretrain.py \
  data_path='[/path/to/lmdb_pretrain]'
```

### 3.2 Fine-tuning

```bash
python main_finetune.py \
  train_data_path='[/path/to/lmdb_train]' \
  val_data_path='[/path/to/lmdb_val]' \
  pretrained_mae=/path/to/pretrain_checkpoint.pth
```

### 3.3 Evaluation

Eval via fine-tune script (evaluates `val_data_path`):

```bash
python main_finetune.py \
  train_data_path='[/path/to/lmdb_train]' \
  val_data_path='[/path/to/lmdb_val]' \
  resume=/path/to/finetune_checkpoint.pth \
  eval=true
```

Standalone eval (recommended for benchmark reporting):

```bash
python eval.py \
  resume=/path/to/finetune_checkpoint.pth \
  test_data_path='[/path/to/lmdb_test]'
```
