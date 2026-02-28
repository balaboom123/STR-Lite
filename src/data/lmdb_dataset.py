import io
import os
from collections import OrderedDict
from collections.abc import Sequence

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from src.data.str_transforms import build_str_transform


class _LmdbBase(Dataset):
    """Shared LMDB access logic for normal LMDB datasets."""

    def _init_lmdb(self, root_dir, return_label, max_label_length, max_retry, readahead):
        self.root_dir = root_dir
        self.return_label = return_label
        self.max_label_length = max_label_length
        self.max_retry = max_retry
        self.readahead = bool(readahead)
        self.env = None
        self._num_samples = self._read_num_samples()

    def _read_num_samples(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"LMDB directory not found: {self.root_dir}")
        if not os.path.isfile(os.path.join(self.root_dir, "data.mdb")):
            raise FileNotFoundError(f"LMDB data.mdb not found: {self.root_dir}")

        env = lmdb.open(
            self.root_dir,
            readonly=True,
            lock=False,
            readahead=self.readahead,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            raw = txn.get(b"num-samples")
            if raw is None:
                raise KeyError("num-samples key not found in LMDB")
            num_samples = int(raw.decode("utf-8"))
        env.close()
        return num_samples

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.root_dir,
                readonly=True,
                lock=False,
                readahead=self.readahead,
                meminit=False,
            )
        return self.env

    def _read_sample(self, txn, file_idx, transform, label_encoder=None):
        for offset in range(self.max_retry):
            actual_idx = ((file_idx - 1 + offset) % self._num_samples) + 1
            img_key = f"image-{actual_idx:09d}".encode()
            imgbuf = txn.get(img_key)
            if imgbuf is None:
                continue

            try:
                image = Image.open(io.BytesIO(imgbuf)).convert("RGB")
            except OSError:
                continue

            label = None
            if self.return_label:
                label_key = f"label-{actual_idx:09d}".encode()
                raw_label = txn.get(label_key)
                label = raw_label.decode("utf-8") if raw_label else ""
                if len(label) > self.max_label_length:
                    continue
                if label_encoder is not None:
                    encoded = label_encoder.encode(label)
                    if encoded is None:
                        continue
                    label = np.asarray(encoded, dtype=np.int16)

            image = transform(image)
            return image, label if self.return_label else None

        return None

    def __len__(self):
        return self._num_samples


class LmdbDataset(_LmdbBase):
    def __init__(
        self,
        root_dir,
        img_height=32,
        img_width=128,
        return_label=True,
        max_label_length=25,
        label_encoder=None,
        augment=True,
        randaugment_layers=2,
        randaugment_magnitude=5,
        randaugment_prob=1.0,
        max_retry=100,
        readahead=True,
    ):
        self._init_lmdb(root_dir, return_label, max_label_length, max_retry, readahead)
        self.img_height = img_height
        self.img_width = img_width
        self.label_encoder = label_encoder

        self.transform = build_str_transform(
            img_height=img_height,
            img_width=img_width,
            augment=augment,
            randaugment_layers=randaugment_layers,
            randaugment_magnitude=randaugment_magnitude,
            randaugment_prob=randaugment_prob,
        )
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])

    def __getitem__(self, index):
        if index < 0 or index >= self._num_samples:
            raise IndexError(f"Index {index} out of range for {self._num_samples} samples")

        env = self._get_env()
        with env.begin(write=False) as txn:
            result = self._read_sample(txn, index + 1, self.transform, label_encoder=self.label_encoder)

        if result is None:
            raise RuntimeError(f"Failed to read a valid sample from {self.root_dir}")

        return result


def discover_lmdb_dirs(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"LMDB root directory not found: {root_dir}")

    lmdb_dirs = []
    root_mdb = os.path.join(root_dir, "data.mdb")
    if os.path.isfile(root_mdb):
        lmdb_dirs.append(root_dir)

    # Recursive scan (including symlinked subfolders) to collect all normal LMDB roots.
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        if "data.mdb" in filenames:
            lmdb_dirs.append(dirpath)

    lmdb_dirs = sorted(set(os.path.normpath(d) for d in lmdb_dirs))
    if not lmdb_dirs:
        raise FileNotFoundError(f"No LMDB datasets found under {root_dir}")
    return lmdb_dirs


def _normalize_root_dirs(root_dir):
    if isinstance(root_dir, str):
        return [root_dir]
    if isinstance(root_dir, Sequence):
        return [str(p) for p in root_dir]
    raise TypeError(f"root_dir must be str or list[str], got {type(root_dir).__name__}")


def _build_single_lmdb_dataset(
    lmdb_dir,
    img_height=32,
    img_width=128,
    return_label=True,
    max_label_length=25,
    label_encoder=None,
    augment=True,
    randaugment_layers=2,
    randaugment_magnitude=5,
    randaugment_prob=1.0,
    readahead=True,
):
    return LmdbDataset(
        root_dir=lmdb_dir,
        img_height=img_height,
        img_width=img_width,
        return_label=return_label,
        max_label_length=max_label_length,
        label_encoder=label_encoder,
        augment=augment,
        randaugment_layers=randaugment_layers,
        randaugment_magnitude=randaugment_magnitude,
        randaugment_prob=randaugment_prob,
        readahead=readahead,
    )


def build_lmdb_datasets_by_name(
    root_dir,
    img_height=32,
    img_width=128,
    return_label=True,
    max_label_length=25,
    label_encoder=None,
    augment=True,
    randaugment_layers=2,
    randaugment_magnitude=5,
    randaugment_prob=1.0,
    readahead=True,
):
    root_dirs = _normalize_root_dirs(root_dir)
    multi_root = len(root_dirs) > 1

    datasets = OrderedDict()
    for single_root in root_dirs:
        lmdb_dirs = discover_lmdb_dirs(single_root)
        root_name = os.path.basename(os.path.normpath(single_root)) or "root"

        for lmdb_dir in lmdb_dirs:
            rel_path = os.path.relpath(lmdb_dir, single_root)
            dataset_name = root_name if rel_path in (".", "") else rel_path.replace(os.sep, "/")
            if multi_root:
                dataset_name = f"{root_name}/{dataset_name}" if rel_path not in (".", "") else root_name

            key = dataset_name
            suffix = 2
            while key in datasets:
                key = f"{dataset_name}_{suffix}"
                suffix += 1

            datasets[key] = _build_single_lmdb_dataset(
                lmdb_dir=lmdb_dir,
                img_height=img_height,
                img_width=img_width,
                return_label=return_label,
                max_label_length=max_label_length,
                label_encoder=label_encoder,
                augment=augment,
                randaugment_layers=randaugment_layers,
                randaugment_magnitude=randaugment_magnitude,
                randaugment_prob=randaugment_prob,
                readahead=readahead,
            )

    return datasets


def build_lmdb_dataset(
    root_dir,
    img_height=32,
    img_width=128,
    return_label=True,
    max_label_length=25,
    label_encoder=None,
    augment=True,
    randaugment_layers=2,
    randaugment_magnitude=5,
    randaugment_prob=1.0,
    readahead=True,
):
    datasets_by_name = build_lmdb_datasets_by_name(
        root_dir=root_dir,
        img_height=img_height,
        img_width=img_width,
        return_label=return_label,
        max_label_length=max_label_length,
        label_encoder=label_encoder,
        augment=augment,
        randaugment_layers=randaugment_layers,
        randaugment_magnitude=randaugment_magnitude,
        randaugment_prob=randaugment_prob,
        readahead=readahead,
    )
    datasets = list(datasets_by_name.values())
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def lmdb_collate_fn(batch):
    if not batch:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8), None

    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    if all(label is None for label in labels):
        return images, None

    if isinstance(labels[0], np.ndarray):
        lengths = torch.tensor([int(l.shape[0]) for l in labels], dtype=torch.int16)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        label_ids = torch.full((len(labels), max_len), -1, dtype=torch.int16)
        for i, arr in enumerate(labels):
            if arr.size > 0:
                label_ids[i, : arr.shape[0]] = torch.from_numpy(arr.astype(np.int16, copy=False))
        return images, {"text": label_ids, "lengths": lengths}

    labels = [label if label is not None else "" for label in labels]
    return images, labels
