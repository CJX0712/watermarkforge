"""Synthetic next-token generator + keyed watermark crypto + attacks.

The DGP is a controlled i.i.d. uniform next-token distribution. This isolates
the *watermark statistics* from LM-quality confounds and matches the theoretical
analysis in Kirchenbauer et al. (2023) and Aaronson/Greenblatt's Gumbel scheme,
where the null (unwatermarked) green fraction is exactly ``gamma``.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from core.config import Config
from core.errors import DataError


class KeyedHasher:
    """Deterministic, keyed green-list + secret-Gumbel primitives, cached by prev token.

    The detector recomputes these from the *token sequence + secret key only*,
    never from the generator's internal draws -- that is the whole point of a
    watermark.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.V = cfg.vocab_size
        self.gamma = cfg.gamma
        self.key = cfg.secret_key
        self._green: Dict[int, np.ndarray] = {}
        self._gumbel: Dict[int, np.ndarray] = {}
        self._green_mat: "np.ndarray | None" = None
        self._gumbel_mat: "np.ndarray | None" = None

    def green_indices(self, prev: int) -> np.ndarray:
        prev = int(prev)
        cached = self._green.get(prev)
        if cached is None:
            h = (prev * 2654435761 + self.key * 40503) & 0xFFFFFFFF
            local = np.random.default_rng(h)
            k = self.cfg.green_size
            cached = local.choice(self.V, size=k, replace=False)
            self._green[prev] = cached
        return cached

    def secret_gumbel(self, prev: int) -> np.ndarray:
        prev = int(prev)
        cached = self._gumbel.get(prev)
        if cached is None:
            h = (prev * 668265263 + self.key * 2246822519) & 0xFFFFFFFF
            cached = np.random.default_rng(h).gumbel(size=self.V)
            self._gumbel[prev] = cached
        return cached

    def ensure_matrices(self) -> None:
        """Lazily build (V, V) green-membership + secret-Gumbel lookup tables."""
        if self._green_mat is None:
            V = self.V
            green = np.zeros((V, V), dtype=bool)
            gumb = np.empty((V, V), dtype=np.float64)
            for a in range(V):
                green[a, self.green_indices(a)] = True
                gumb[a] = self.secret_gumbel(a)
            self._green_mat = green
            self._gumbel_mat = gumb


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one token per row from a (n, V) probability matrix."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return (u > cdf).sum(axis=1).astype(np.int64)


class SyntheticGenerator:
    """Produces unwatermarked and watermarked token matrices + attack transforms."""

    def __init__(self, cfg: Config, hasher: KeyedHasher) -> None:
        self.cfg = cfg
        self.V = cfg.vocab_size
        self.hasher = hasher
        # per-scheme strength scalars; ``wmfuse`` scales both fused channels
        self.strengths: Dict[str, float] = {
            "kgw": cfg.kgw_delta, "gumbel": cfg.gumbel_beta, "wmfuse": 1.0,
        }

    def set_strength(self, scheme: str, value: float) -> None:
        if scheme not in self.strengths:
            raise DataError(f"unknown scheme {scheme!r}")
        self.strengths[scheme] = float(value)

    # ---- scheme logit builders ------------------------------------------------
    def kgw_logits(self, prevs: np.ndarray, delta: float) -> np.ndarray:
        self.hasher.ensure_matrices()
        green = self.hasher._green_mat[prevs]  # (n_seq, V) bool
        return np.where(green, delta, 0.0)

    def gumbel_logits(self, prevs: np.ndarray, beta: float) -> np.ndarray:
        self.hasher.ensure_matrices()
        return beta * self.hasher._gumbel_mat[prevs]  # (n_seq, V)

    def wmfuse_logits(self, prevs: np.ndarray, green_delta: float, gumbel_beta: float) -> np.ndarray:
        self.hasher.ensure_matrices()
        green = self.hasher._green_mat[prevs]
        return gumbel_beta * self.hasher._gumbel_mat[prevs] + np.where(green, green_delta, 0.0)

    # ---- sequence generation --------------------------------------------------
    def _gen_chain(self, n_seq: int, length: int, rng: np.random.Generator,
                   logits_fn) -> np.ndarray:
        xs = np.zeros((n_seq, length), dtype=np.int64)
        prevs = np.full(n_seq, self.cfg.start_token, dtype=np.int64)
        for t in range(length):
            logits = logits_fn(prevs)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            x = _sample_categorical(probs, rng)
            xs[:, t] = x
            prevs = x
        return xs

    def sample_unwatermarked(self, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.V, size=(n_seq, length), dtype=np.int64)

    def watermark(self, scheme: str, n_seq: int, length: int, rng: np.random.Generator) -> np.ndarray:
        s = self.strengths[scheme]
        if scheme == "kgw":
            return self._gen_chain(n_seq, length, rng, lambda pv: self.kgw_logits(pv, s))
        if scheme == "gumbel":
            return self._gen_chain(n_seq, length, rng, lambda pv: self.gumbel_logits(pv, s))
        if scheme == "wmfuse":
            gd = s * self.cfg.wmfuse_green_delta
            gb = s * self.cfg.wmfuse_gumbel_beta
            return self._gen_chain(n_seq, length, rng, lambda pv: self.wmfuse_logits(pv, gd, gb))
        raise DataError(f"unknown scheme {scheme!r}")

    # ---- attacks (simulate paraphrase / rewrite) ------------------------------
    def apply_random_attack(self, xs: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Each token independently replaced by a uniform-random token with prob ``rate``."""
        if not 0.0 <= rate < 1.0:
            raise DataError("attack rate must be in [0, 1)")
        mask = rng.random(xs.shape) < rate
        return np.where(mask, rng.integers(0, self.V, size=xs.shape, dtype=np.int64), xs)

    def apply_contiguous_attack(self, xs: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """A contiguous middle span of ``rate * length`` tokens replaced by uniform-random."""
        if not 0.0 <= rate < 1.0:
            raise DataError("attack rate must be in [0, 1)")
        n_seq, length = xs.shape
        out = xs.copy()
        span = max(1, int(round(rate * length)))
        start = (length - span) // 2
        out[:, start:start + span] = rng.integers(0, self.V, size=(n_seq, span), dtype=np.int64)
        return out

    def apply_green_targeted_attack(self, xs: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Adversary aware of the green-list channel: green tokens are replaced with
        uniform-random ones with probability ``rate``. This *selectively destroys*
        the green indicator -- but only the ``gamma``-fraction of positions that are
        green ever carried KGW signal, so a channel-diverse watermark keeps the rest."""
        if not 0.0 <= rate < 1.0:
            raise DataError("attack rate must be in [0, 1)")
        self.hasher.ensure_matrices()
        n_seq, length = xs.shape
        out = xs.copy()
        prevs = np.full(n_seq, self.cfg.start_token, dtype=np.int64)
        green_mat = self.hasher._green_mat
        for t in range(length):
            is_green = green_mat[prevs, xs[:, t]]
            hit = (rng.random(n_seq) < rate) & is_green
            out[hit, t] = rng.integers(0, self.V, size=int(hit.sum()), dtype=np.int64)
            prevs = xs[:, t]
        return out

    def apply_gumbel_targeted_attack(self, xs: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Adversary aware of the Gumbel channel: tokens whose centered secret-Gumbel
        score is positive (the high-signal half of the vocabulary) are replaced with
        uniform-random ones with probability ``rate``. A Gumbel-only watermark dies to
        this; a channel-diverse watermark survives via its green indicator."""
        if not 0.0 <= rate < 1.0:
            raise DataError("attack rate must be in [0, 1)")
        self.hasher.ensure_matrices()
        n_seq, length = xs.shape
        out = xs.copy()
        prevs = np.full(n_seq, self.cfg.start_token, dtype=np.int64)
        gumb = self.hasher._gumbel_mat
        gmean = gumb.mean(axis=1)
        for t in range(length):
            g = gumb[prevs, xs[:, t]] - gmean[prevs]
            hit = (rng.random(n_seq) < rate) & (g > 0.0)
            out[hit, t] = rng.integers(0, self.V, size=int(hit.sum()), dtype=np.int64)
            prevs = xs[:, t]
        return out
