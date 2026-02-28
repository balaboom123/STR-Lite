from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence


class LabelEncode:
    """Convert text labels to charset indices with OCR-style filtering semantics."""

    def __init__(
        self,
        max_text_length: int,
        character_dict_path: Optional[str] = None,
        use_space_char: bool = False,
        lower: bool = False,
        charset: Optional[Iterable[str]] = None,
    ) -> None:
        if max_text_length <= 0:
            raise ValueError(f"max_text_length must be positive, got {max_text_length}")
        self.max_text_len = int(max_text_length)
        self.lower = bool(lower)

        dict_character = None
        if character_dict_path is not None:
            loaded_chars = []
            with open(character_dict_path, "rb") as fin:
                for raw_line in fin.readlines():
                    line = raw_line.decode("utf-8").strip("\n").strip("\r\n")
                    if line:
                        loaded_chars.append(line)
            if use_space_char:
                loaded_chars.append(" ")
            dict_character = loaded_chars
        elif charset is not None:
            base_chars = list("".join(charset))
            if use_space_char and " " not in base_chars:
                base_chars.append(" ")
            dict_character = base_chars
        else:
            warnings.warn(
                "character_dict_path and charset are both None; fallback charset is digits+lowercase",
                stacklevel=2,
            )
            dict_character = list("0123456789abcdefghijklmnopqrstuvwxyz")
            self.lower = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: idx for idx, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character: Sequence[str]) -> Sequence[str]:
        return list(dict_character)

    def encode(self, text: Optional[str]) -> Optional[list[int]]:
        """BaseRec behavior: skip OOV chars; drop sample if empty or too long."""
        if text is None or len(text) == 0:
            return None
        if self.lower:
            text = text.lower()

        text_ids = []
        for char in text:
            if char not in self.dict:
                continue
            text_ids.append(self.dict[char])

        if len(text_ids) == 0 or len(text_ids) > self.max_text_len:
            return None
        return text_ids
