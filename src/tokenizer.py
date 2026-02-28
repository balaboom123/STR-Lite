from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from src.label_encoder import LabelEncode


DEFAULT_CHAR_DICT = "util/EN_symbol_dict.txt"


class CharsetTokenizer:
    """Autoregressive tokenizer with BOS/EOS/PAD and dict-backed charset ids."""

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

        self.character = list(self.label_encoder.character)
        self.lower = bool(lower)

        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

        self.char_offset = 3
        self.vocab_size = len(self.character) + self.char_offset

        # Max decoder steps for teacher forcing sequence (BOS+chars and chars+EOS)
        self.max_text_length = int(max_text_length)
        self.max_seq_len = self.max_text_length + 1

    def normalize(self, text: str | None) -> str:
        if text is None:
            return ""
        return text.lower() if self.lower else text

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.normalize(t) for t in texts]

    def build_decoder_inputs_from_text_ids(
        self,
        text_ids: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build AR decoder input/target from charset ids.

        Input:
          text_ids: int16/long [B, T_char] in charset-index space [0..charset-1]
          lengths: int16/long [B]
        Output (on `device`):
          input_ids: long [B, max_seq_len]
          target_ids: long [B, max_seq_len]
          key_padding_mask: bool [B, max_seq_len] (True for PAD)
        """
        if text_ids.ndim != 2:
            raise ValueError(f"text_ids must be [B, T], got {tuple(text_ids.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)}")
        if text_ids.size(0) != lengths.size(0):
            raise ValueError("Batch mismatch between text_ids and lengths")

        bsz = text_ids.size(0)
        text_ids = text_ids.to(device=device, non_blocking=True)
        lengths = lengths.to(device=device, non_blocking=True, dtype=torch.long)

        input_ids = torch.full((bsz, self.max_seq_len), self.pad_id, dtype=torch.long, device=device)
        target_ids = torch.full((bsz, self.max_seq_len), self.pad_id, dtype=torch.long, device=device)

        for i in range(bsz):
            cur_len = int(lengths[i].item())
            cur_len = max(0, min(cur_len, self.max_text_length))

            token_ids = text_ids[i, :cur_len].to(dtype=torch.long) + self.char_offset

            in_seq = torch.cat([
                torch.tensor([self.bos_id], device=device, dtype=torch.long),
                token_ids,
            ])
            tgt_seq = torch.cat([
                token_ids,
                torch.tensor([self.eos_id], device=device, dtype=torch.long),
            ])

            seq_len = min(in_seq.numel(), self.max_seq_len)
            input_ids[i, :seq_len] = in_seq[:seq_len]
            target_ids[i, :seq_len] = tgt_seq[:seq_len]

        key_padding_mask = input_ids.eq(self.pad_id)
        return input_ids, target_ids, key_padding_mask

    def decode_token_ids(self, token_ids: Sequence[int]) -> str:
        chars: List[str] = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (self.pad_id, self.bos_id):
                continue
            if tid == self.eos_id:
                break
            cid = tid - self.char_offset
            if 0 <= cid < len(self.character):
                chars.append(self.character[cid])
        return "".join(chars)

    def decode_token_ids_batch(self, token_ids: torch.Tensor) -> List[str]:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be [B, T], got {tuple(token_ids.shape)}")
        token_ids = token_ids.detach().cpu()
        return [self.decode_token_ids(row.tolist()) for row in token_ids]

    def decode_char_ids_batch(self, text_ids: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        if text_ids.ndim != 2:
            raise ValueError(f"text_ids must be [B, T], got {tuple(text_ids.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)}")
        if text_ids.size(0) != lengths.size(0):
            raise ValueError("Batch mismatch between text_ids and lengths")

        text_ids = text_ids.detach().cpu()
        lengths = lengths.detach().cpu()

        out: List[str] = []
        for i in range(text_ids.size(0)):
            cur_len = int(lengths[i].item())
            cur_len = max(0, min(cur_len, text_ids.size(1)))
            chars: List[str] = []
            for cid in text_ids[i, :cur_len].tolist():
                cid = int(cid)
                if 0 <= cid < len(self.character):
                    chars.append(self.character[cid])
            out.append("".join(chars))
        return out
