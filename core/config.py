"""Configuration with ENV overrides + schema validation.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from core.errors import ConfigError

ENV_PREFIX = "WM_"


@dataclass
class Config:
    """All knobs for one benchmark run.

    Override any field via environment variable ``WM_<UPPER_FIELD>``.
    """

    vocab_size: int = 2000
    gamma: float = 0.5  # green-list fraction
    secret_key: int = 20261010
    seed: int = 7
    start_token: int = 0  # context token preceding position 0
    n_seq: int = 100  # sequences per condition
    lengths: List[int] = field(default_factory=lambda: [30, 50, 100, 200])
    kgw_delta: float = 2.0  # KGW green-logit boost (overridden if equal_distortion)
    gumbel_beta: float = 1.0  # Gumbel distortion-free tilt scale (overridden if equal_distortion)
    wmfuse_green_delta: float = 1.318  # WmFuse green boost at balanced split
    wmfuse_gumbel_beta: float = 0.32  # WmFuse Gumbel tilt at balanced split
    window_scales: List[int] = field(default_factory=lambda: [8, 16, 32])  # multi-scale localization
    target_kl: float = 0.30  # equal-distortion budget (nats/token) for fair comparison
    equal_distortion: bool = True  # calibrate every scheme to target_kl before scoring
    fpr_target: float = 0.01  # operating point for TPR@FPR
    attack_rates: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    attacks: List[str] = field(default_factory=lambda: [
        "uniform", "contiguous", "green_targeted", "gumbel_targeted"])
    seeds: List[int] = field(default_factory=lambda: [7, 11, 23])

    def __post_init__(self) -> None:
        self._apply_env()
        self._validate()

    def _apply_env(self) -> None:
        mapping = {
            "WM_VOCAB_SIZE": ("vocab_size", int),
            "WM_GAMMA": ("gamma", float),
            "WM_SECRET_KEY": ("secret_key", int),
            "WM_SEED": ("seed", int),
            "WM_N_SEQ": ("n_seq", int),
            "WM_KGW_DELTA": ("kgw_delta", float),
            "WM_GUMBEL_BETA": ("gumbel_beta", float),
            "WM_FPR_TARGET": ("fpr_target", float),
        }
        for env, (attr, caster) in mapping.items():
            val = os.environ.get(env)
            if val is not None:
                try:
                    setattr(self, attr, caster(val))
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ConfigError(f"{env}={val!r} not castable to {caster.__name__}") from exc

    def _validate(self) -> None:
        if self.vocab_size < 16:
            raise ConfigError("vocab_size must be >= 16")
        if not 0.0 < self.gamma < 1.0:
            raise ConfigError("gamma must be in (0, 1)")
        if self.n_seq < 8:
            raise ConfigError("n_seq must be >= 8")
        if not all(ln >= 2 for ln in self.lengths):
            raise ConfigError("all lengths must be >= 2")
        if self.kgw_delta <= 0:
            raise ConfigError("kgw_delta must be > 0")
        if self.gumbel_beta <= 0:
            raise ConfigError("gumbel_beta must be > 0")
        if not 0.0 < self.fpr_target < 1.0:
            raise ConfigError("fpr_target must be in (0, 1)")

    @property
    def green_size(self) -> int:
        return int(round(self.gamma * self.vocab_size))
