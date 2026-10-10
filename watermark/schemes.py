"""Watermark injection schemes (Protocol-compliant, single-sequence API for CLI/demo).

The matrix-level benchmark in ``eval/benchmark.py`` calls the generator directly
for throughput; these classes expose the same schemes through the
``WatermarkScheme`` contract for one sequence at a time.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.interfaces import WatermarkScheme
from core.types import TokenSeq, WatermarkResult
from data.generator import KeyedHasher, SyntheticGenerator


class _BaseScheme:
    name = "base"

    def __init__(self, cfg: Config, hasher: KeyedHasher, gen: SyntheticGenerator) -> None:
        self.cfg = cfg
        self.hasher = hasher
        self.gen = gen

    def _strength(self) -> float:
        raise NotImplementedError

    def watermark(self, seq: TokenSeq, rng: np.random.Generator) -> WatermarkResult:
        length = len(seq.tokens)
        mat = self.gen.watermark(self.name, 1, length, rng)
        tokens = [int(t) for t in mat[0]]
        scores = self._per_token_score(tokens)
        kl = self._kl_from_tokens(tokens)
        return WatermarkResult(
            tokens=tokens, scheme=self.name, secret_key=self.cfg.secret_key,
            gamma=self.cfg.gamma, strength=self._strength(),
            per_token_score=scores, kl=kl,
        )

    def _per_token_score(self, tokens):  # pragma: no cover - overridden
        return [0.0] * len(tokens)

    def _kl_from_tokens(self, tokens):
        V = self.cfg.vocab_size
        counts = np.bincount(np.asarray(tokens, dtype=int), minlength=V).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(np.sum(p * (np.log(p) + np.log(V))))


class KGWScheme(_BaseScheme):
    name = "kgw"

    def _strength(self) -> float:
        return self.cfg.kgw_delta

    def _per_token_score(self, tokens):
        out = []
        prev = self.cfg.start_token
        for x in tokens:
            out.append(1.0 if np.isin(x, self.hasher.green_indices(int(prev))) else 0.0)
            prev = x
        return out


class GumbelScheme(_BaseScheme):
    name = "gumbel"

    def _strength(self) -> float:
        return self.cfg.gumbel_beta

    def _per_token_score(self, tokens):
        self.hasher.ensure_matrices()
        gmean = self.hasher._gumbel_mat.mean(axis=1)
        out = []
        prev = self.cfg.start_token
        for x in tokens:
            out.append(float((self.hasher._gumbel_mat[int(prev), x] - gmean[int(prev)])
                             / np.sqrt(np.pi ** 2 / 6.0)))
            prev = x
        return out


class WmFuseScheme(_BaseScheme):
    name = "wmfuse"

    def _strength(self) -> float:
        return self.cfg.wmfuse_gumbel_beta

    def _per_token_score(self, tokens):
        self.hasher.ensure_matrices()
        gmean = self.hasher._gumbel_mat.mean(axis=1)
        gum, green = [], []
        prev = self.cfg.start_token
        for x in tokens:
            g = float(self.hasher._gumbel_mat[int(prev), x] - gmean[int(prev)])
            gum.append(g / np.sqrt(np.pi ** 2 / 6.0))
            green.append(1.0 if self.hasher._green_mat[int(prev), x] else 0.0)
            prev = x
        g_std = float(np.sqrt(np.pi ** 2 / 6.0))
        k_std = float(np.sqrt(self.cfg.gamma * (1.0 - self.cfg.gamma)))
        return [(g / g_std + (gr - self.cfg.gamma) / k_std) / np.sqrt(2.0)
                for g, gr in zip(gum, green)]


_SCHEMES = {"kgw": KGWScheme, "gumbel": GumbelScheme, "wmfuse": WmFuseScheme}


def get_scheme(name: str, cfg: Config, hasher: KeyedHasher, gen: SyntheticGenerator) -> WatermarkScheme:
    if name not in _SCHEMES:
        raise KeyError(f"no scheme {name!r}")
    return _SCHEMES[name](cfg, hasher, gen)
