# Installation

## Requirements

- Python 3.10+ (recommended)
- CUDA GPU (recommended for training)

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Verify Environment

```bash
python -c "import torch; print(torch.__version__)"
python -c "import hydra; print(hydra.__version__)"
```

## Notes

- This project uses Hydra for config-driven runs.
- If you use multiple GPUs, launch with `torchrun`.
- Default precision in configs is `bf16`; ensure your hardware supports it.
