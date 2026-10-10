"""Equal-distortion calibration.

Comparing a green-list watermark to a per-token score watermark is only fair when
the *text distortion* they impose is matched. That distortion is the exact
per-token KL between the watermarked next-token distribution and the uniform base
distribution; sampling it (empirical KL) carries a (V-1)/(2N) estimation bias that
dominates at realistic sample sizes, so we compute it analytically.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.errors import EvalError
from data.generator import KeyedHasher

# Deterministically subsampled context tokens used to average the per-context KL.
_N_CTX = 256


def _contexts(cfg: Config) -> np.ndarray:
    return np.linspace(0, cfg.vocab_size - 1, num=min(_N_CTX, cfg.vocab_size)).astype(int)


def _kl_from_logits(logits: np.ndarray, V: int) -> float:
    z = logits - logits.max()
    p = np.exp(z)
    p = p / p.sum()
    nz = p[p > 0]
    H = -float(np.sum(nz * np.log(nz)))
    return float(np.log(V) - H)


def scheme_logits(cfg: Config, hasher: KeyedHasher, scheme: str, strength: float,
                  prev: int) -> np.ndarray:
    """Watermarked next-token logits at context ``prev`` for a given strength scalar."""
    hasher.ensure_matrices()
    if scheme == "kgw":
        green = hasher._green_mat[prev]
        return np.where(green, strength, 0.0)
    if scheme == "gumbel":
        return strength * hasher._gumbel_mat[prev]
    if scheme == "wmfuse":
        green = hasher._green_mat[prev]
        return (strength * cfg.wmfuse_gumbel_beta * hasher._gumbel_mat[prev]
                + np.where(green, strength * cfg.wmfuse_green_delta, 0.0))
    raise EvalError(f"unknown scheme {scheme!r}")


def distortion(cfg: Config, hasher: KeyedHasher, scheme: str, strength: float) -> float:
    """Exact per-token distortion (KL from uniform), averaged over contexts."""
    V = cfg.vocab_size
    if strength <= 0:
        return 0.0
    kls = [_kl_from_logits(scheme_logits(cfg, hasher, scheme, strength, int(a)), V)
           for a in _contexts(cfg)]
    return float(np.mean(kls))


def calibrate_strength(cfg: Config, hasher: KeyedHasher, scheme: str, target_kl: float,
                       lo: float = 1e-4, hi: float = 50.0, iters: int = 44) -> float:
    """Bisect the strength scalar so that the scheme's distortion == target_kl."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if distortion(cfg, hasher, scheme, mid) < target_kl:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
