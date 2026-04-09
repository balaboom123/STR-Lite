# Dataset Format

STRLite expects LMDB datasets with OCR-style key naming.

## LMDB Schema

Each LMDB must include:

- `num-samples`
- `image-000000001`, `image-000000002`, ...
- `label-000000001`, `label-000000002`, ... (required for fine-tuning and evaluation)

## Path Inputs

- `data_path`: pretraining dataset root(s)
- `train_data_path`: fine-tuning training root(s)
- `val_data_path`: fine-tuning validation root(s)
- `test_data_path`: standalone evaluation root(s)

Each path can be either:

- a single path string
- a list of path strings

The loader recursively discovers nested folders containing `data.mdb`.

## Script Usage

- `main_pretrain.py` uses `data_path`
- `main_finetune.py` uses `train_data_path` + `val_data_path`
- `main_finetune.py eval=true` evaluates on `val_data_path`
- `eval.py` evaluates on `test_data_path`

## Common Issues

- Missing `num-samples` key -> dataset load error
- Missing `label-*` keys -> fine-tuning/evaluation cannot build decoder targets
- Character mismatch with dictionary (`util/EN_symbol_dict.txt`) -> labels may be filtered
