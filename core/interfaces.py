"""Protocol interfaces (single source of contract).

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import DetectionResult, TokenSeq, WatermarkResult


@runtime_checkable
class WatermarkScheme(Protocol):
    """A watermark injection scheme."""

    name: str

    def watermark(self, seq: TokenSeq, rng: object) -> WatermarkResult:
        """Return a watermarked version of ``seq``."""
        ...


@runtime_checkable
class Detector(Protocol):
    """A watermark detector (holds the secret key)."""

    name: str

    def detect(self, seq: TokenSeq) -> DetectionResult:
        """Score ``seq``; higher score => more likely watermarked."""
        ...
