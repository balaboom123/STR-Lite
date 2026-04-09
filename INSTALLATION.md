# Installation Guide

This document provides step-by-step installation instructions for STRLite.
It is recommended to complete this setup once before running training.

## 1. Requirements

- Python 3.10+
- NVIDIA GPU is recommended for training
- Windows / Linux are both supported

## 2. Create a Virtual Environment (Recommended)

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install Dependencies

Run in the project root:

```bash
pip install -r requirements.txt
```

## 4. Verify Installation

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import hydra; print('hydra:', hydra.__version__)"
python -c "import timm, lmdb, numpy, PIL; print('deps ok')"
```

If you have a GPU, also check CUDA availability:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

## 5. First Runnable Check

Run these to verify the script entrypoints:

```bash
python main_pretrain.py --help
python main_finetune.py --help
python eval.py --help
```
