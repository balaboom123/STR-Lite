# Dataset Guide

This document explains two things:

1. How to obtain datasets
2. How to organize them into a format directly usable by STRLite

## 1. Available Data Sources

You can use either of the following:

1. Your own OCR/STR dataset (most common)
2. Public STR datasets (for example, Union14M or other benchmarks)

If you want to use Union14M as a reference, use the official repository:

- https://github.com/Mountchicken/Union14M

Note: This project does not include built-in download scripts.
Prepare your dataset first, then pass the dataset paths to the training/evaluation scripts.

## 2. LMDB Format Required by STRLite

Each LMDB folder should contain at least:

- `data.mdb`
- `lock.mdb`
- key: `num-samples`
- key: `image-000000001`, `image-000000002`, ...
- key: `label-000000001`, `label-000000002`, ... (labels are optional for pretraining)

Here, `label-*` stores text annotations, and `num-samples` stores the total sample count.

## 3. Path Parameter Mapping

- `data_path`: pretraining data (`main_pretrain.py`)
- `train_data_path`: fine-tuning training data (`main_finetune.py`)
- `val_data_path`: fine-tuning validation data (`main_finetune.py`)
- `test_data_path`: standalone test data (`eval.py`)

Each path parameter can be:

1. A single path string
2. A list of paths

The loader recursively discovers folders that contain `data.mdb`.

## 4. Recommended Directory Layout

```text
datasets/
	pretrain/
		unlabeled_lmdb_1/
		unlabeled_lmdb_2/
	finetune/
		train/
			lmdb_a/
			lmdb_b/
		val/
			lmdb_val/
		test/
			lmdb_test/
```

## 5. Practical Usage

### 5.1 Pretraining (Images Only)

```bash
python main_pretrain.py data_path='[/path/to/datasets/pretrain]'
```

### 5.2 Fine-tuning (Images + Labels)

```bash
python main_finetune.py \
	train_data_path='[/path/to/datasets/finetune/train]' \
	val_data_path='[/path/to/datasets/finetune/val]' \
	pretrained_mae=/path/to/pretrain_checkpoint.pth
```

### 5.3 Final Test (Recommended via eval.py)

```bash
python eval.py \
	resume=/path/to/finetune_checkpoint.pth \
	test_data_path='[/path/to/datasets/finetune/test]'
```

## 6. Common Issues

1. Missing `num-samples`

- Symptom: Dataset loading fails
- Fix: Ensure the LMDB contains the `num-samples` key

2. Missing `label-*`

- Symptom: Fine-tuning or evaluation fails
- Fix: Ensure train/val/test LMDB datasets contain label keys

3. Characters are filtered out

- Symptom: Fewer usable samples or empty labels
- Fix: Check whether `util/EN_symbol_dict.txt` covers your annotation charset

4. Wrong eval split used

- `main_finetune.py eval=true` evaluates `val_data_path`
- For final test evaluation, use `eval.py` with `test_data_path`
