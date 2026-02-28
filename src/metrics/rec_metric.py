from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _safe_text(text: str | None) -> str:
    return text or ""


@dataclass
class _Counts:
    total: int = 0
    correct_num_real: int = 0
    correct_num_lower: int = 0
    correct_num: int = 0


class RecMetric:
    """Word-level recognition metrics aligned with OCR normalization semantics."""

    def __init__(
        self,
        main_indicator: str = "acc",
        is_filter: bool = True,
        is_lower: bool = True,
        ignore_space: bool = True,
        valid_chars: Iterable[str] | None = None,
        remove_symbols: bool = True,
    ) -> None:
        self.main_indicator = main_indicator
        self.is_filter = bool(is_filter)
        self.is_lower = bool(is_lower)
        self.ignore_space = bool(ignore_space)
        self.remove_symbols = bool(remove_symbols)
        self.valid_chars = set(valid_chars) if valid_chars is not None else None
        self._counts = _Counts()

    def reset(self) -> None:
        self._counts = _Counts()

    def _normalize_for_main(self, text: str) -> str:
        result = text
        if self.ignore_space:
            result = "".join(ch for ch in result if not ch.isspace())
        if self.is_filter and self.valid_chars is not None:
            result = "".join(ch for ch in result if ch in self.valid_chars)
        if self.remove_symbols:
            result = "".join(ch for ch in result if ch.isalnum() or ord(ch) > 127)
        if self.is_lower:
            result = result.lower()
        return result

    def update(self, pred: str | None, target: str | None) -> None:
        pred_raw = _safe_text(pred)
        target_raw = _safe_text(target)
        self._counts.total += 1

        if pred_raw == target_raw:
            self._counts.correct_num_real += 1
        if pred_raw.lower() == target_raw.lower():
            self._counts.correct_num_lower += 1

        pred_main = self._normalize_for_main(pred_raw)
        target_main = self._normalize_for_main(target_raw)
        if pred_main == target_main:
            self._counts.correct_num += 1

    def update_many(self, preds: Iterable[str], targets: Iterable[str]) -> None:
        for pred, target in zip(preds, targets):
            self.update(pred, target)

    def get_counts(self) -> dict[str, int]:
        return {
            "total": self._counts.total,
            "correct_num_real": self._counts.correct_num_real,
            "correct_num_lower": self._counts.correct_num_lower,
            "correct_num": self._counts.correct_num,
        }

    @staticmethod
    def _to_rate(correct: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return float(correct) / float(total)

    def get_metric(self) -> dict[str, float | int]:
        total = self._counts.total
        acc_real = self._to_rate(self._counts.correct_num_real, total)
        acc_lower = self._to_rate(self._counts.correct_num_lower, total)
        acc = self._to_rate(self._counts.correct_num, total)
        return {
            "acc": acc,
            "acc_real": acc_real,
            "acc_lower": acc_lower,
            "total": total,
            "correct": self._counts.correct_num,
            "correct_num_real": self._counts.correct_num_real,
            "correct_num_lower": self._counts.correct_num_lower,
            "correct_num": self._counts.correct_num,
        }
