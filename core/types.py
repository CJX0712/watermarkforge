"""Core datatypes.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TokenSeq:
    """A token sequence plus the context token preceding its first element.

    `prev` is what the green-list hash keys off for position 0.
    """

    tokens: List[int]
    prev: int = 0


@dataclass
class WatermarkResult:
    """Output of a watermark scheme on one sequence."""

    tokens: List[int]
    scheme: str
    secret_key: int
    gamma: float
    strength: float  # delta (KGW) or beta (Gumbel)
    per_token_score: List[float] = field(default_factory=list)
    kl: float = 0.0  # KL(base || watermarked) distortion proxy (mean over tokens)


@dataclass
class DetectionResult:
    """Output of a detector on one sequence."""

    score: float
    is_watermarked: bool
    scheme: str = ""
    z: Optional[float] = None


@dataclass
class BenchmarkRecord:
    """One row of the benchmark table."""

    scheme: str
    detector: str
    length: int
    attack: str
    attack_rate: float
    auc: float
    tpr_at_fpr: float  # TPR @ fpr_target
    mean_kl: float
    n_seq: int
    seed: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "scheme": self.scheme,
            "detector": self.detector,
            "length": self.length,
            "attack": self.attack,
            "attack_rate": self.attack_rate,
            "auc": self.auc,
            "tpr_at_fpr": self.tpr_at_fpr,
            "mean_kl": self.mean_kl,
            "n_seq": self.n_seq,
            "seed": self.seed,
        }
