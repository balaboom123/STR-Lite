from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from src.label_encoder import LabelEncode


DEFAULT_CHAR_DICT = "util/EN_symbol_dict.txt"


class CTCLabelConverter:
    """CTC converter backed by OCR label encoder and dictionary file."""

    def __init__(
        self,
        character_dict_path: str = DEFAULT_CHAR_DICT,
        max_text_length: int = 25,
        lower: bool = False,
        use_space_char: bool = False,
    ):
        if not Path(character_dict_path).is_file():
            raise FileNotFoundError(f"Character dict file not found: {character_dict_path}")

        self.label_encoder = LabelEncode(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            lower=lower,
        )

        self.blank_idx = 0
        self.character = list(self.label_encoder.character)
        self.idx_to_char = ["[blank]"] + self.character
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.character)}

        self.lower = lower
        self.max_text_length = max_text_length

    @property
    def num_classes(self):
        return len(self.idx_to_char)

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        return text.lower() if self.lower else text

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.normalize(t) for t in texts]

    def encode(self, texts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        targets: List[int] = []
        lengths: List[int] = []

        for text in texts:
            text_ids = self.label_encoder.encode(text)
            if text_ids is None:
                text_ids = []

            # Shift by +1 to reserve 0 for CTC blank.
            ctc_ids = [tid + 1 for tid in text_ids]
            targets.extend(ctc_ids)
            lengths.append(len(ctc_ids))

        targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        return targets_tensor, lengths_tensor

    def encode_from_text_ids(
        self,
        text_ids: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode int16 charset indices (CPU/GPU) to CTC targets on GPU."""
        if text_ids.ndim != 2:
            raise ValueError(f"text_ids must be [B, T], got shape {tuple(text_ids.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be [B], got shape {tuple(lengths.shape)}")
        if text_ids.size(0) != lengths.size(0):
            raise ValueError("Batch size mismatch between text_ids and lengths")

        text_ids = text_ids.to(device=device, non_blocking=True)
        lengths = lengths.to(device=device, non_blocking=True, dtype=torch.long)

        parts = []
        for i in range(text_ids.size(0)):
            cur_len = int(lengths[i].item())
            if cur_len <= 0:
                continue
            ids_i = text_ids[i, :cur_len].to(dtype=torch.long) + 1  # reserve 0 for CTC blank
            parts.append(ids_i)

        if parts:
            targets = torch.cat(parts, dim=0)
        else:
            targets = torch.empty((0,), dtype=torch.long, device=device)
        return targets, lengths

    def decode_text_ids_batch(self, text_ids: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        """Decode CPU/GPU int16 ids (without CTC blank shift) to raw text list."""
        if text_ids.ndim != 2:
            raise ValueError(f"text_ids must be [B, T], got shape {tuple(text_ids.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be [B], got shape {tuple(lengths.shape)}")
        if text_ids.size(0) != lengths.size(0):
            raise ValueError("Batch size mismatch between text_ids and lengths")

        text_ids = text_ids.detach().cpu()
        lengths = lengths.detach().cpu()

        out: List[str] = []
        for i in range(text_ids.size(0)):
            cur_len = int(lengths[i].item())
            chars: List[str] = []
            for raw_id in text_ids[i, :cur_len].tolist():
                idx = int(raw_id)
                if idx < 0 or idx >= len(self.character):
                    continue
                chars.append(self.character[idx])
            out.append("".join(chars))
        return out

    def decode(self, preds_index, preds_length=None) -> List[str]:
        if isinstance(preds_index, torch.Tensor):
            preds_index = preds_index.detach().cpu().tolist()

        if preds_length is None:
            preds_length = [len(seq) for seq in preds_index]
        elif isinstance(preds_length, torch.Tensor):
            preds_length = preds_length.detach().cpu().tolist()

        texts = []
        for seq, length in zip(preds_index, preds_length):
            seq = seq[:length]
            prev = self.blank_idx
            chars = []
            for idx in seq:
                idx = int(idx)
                if idx != self.blank_idx and idx != prev:
                    if 0 <= idx < len(self.idx_to_char):
                        chars.append(self.idx_to_char[idx])
                prev = idx
            texts.append("".join(chars))
        return texts
