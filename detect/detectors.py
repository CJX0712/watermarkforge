"""Watermark detectors (score-only; higher score => more likely watermarked).

All detectors recompute their signal from the *token sequence + secret key* alone.

Channel centering
-----------------
The secret-Gumbel values ``g_v`` at a context have mean ``E[g]`` over the
vocabulary, not 0. We subtract that per-context mean (a deterministic function of
the key only) so the *unwatermarked* per-token score has mean exactly 0 and the
whole-sequence statistic is a clean standard normal under the null. This makes the
null z-scores interpretable and the fixed 2.33 threshold in ``detect()`` valid.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.interfaces import Detector
from core.types import DetectionResult, TokenSeq
from data.generator import KeyedHasher

_VAR_GUMBEL = np.pi**2 / 6.0  # variance of Gumbel(0, 1)


def _green_per_token(xs: np.ndarray, hasher: KeyedHasher, cfg: Config) -> np.ndarray:
    """Indicator (1 = green) per token, with `start_token` as the position-0 prev."""
    hasher.ensure_matrices()
    n_seq, L = xs.shape
    ind = np.zeros((n_seq, L), dtype=np.float64)
    prevs = np.full(n_seq, cfg.start_token, dtype=np.int64)
    green_mat = hasher._green_mat
    for t in range(L):
        ind[:, t] = green_mat[prevs, xs[:, t]]
        prevs = xs[:, t]
    return ind


def _centered_gumbel_per_token(xs: np.ndarray, hasher: KeyedHasher, cfg: Config) -> np.ndarray:
    """Centered secret-Gumbel value ``g_v - mean_v(g)`` per token.

    Subtracting the per-context mean makes the null per-token mean exactly 0 while
    leaving the variance (hence the watermarked signal-to-noise ratio) unchanged.
    """
    hasher.ensure_matrices()
    n_seq, L = xs.shape
    sc = np.zeros((n_seq, L), dtype=np.float64)
    prevs = np.full(n_seq, cfg.start_token, dtype=np.int64)
    gumbel_mat = hasher._gumbel_mat
    gmean = gumbel_mat.mean(axis=1)
    for t in range(L):
        sc[:, t] = gumbel_mat[prevs, xs[:, t]] - gmean[prevs]
        prevs = xs[:, t]
    return sc


def _windowed_max_z(fused: np.ndarray, window: int, null_std_w: float) -> np.ndarray:
    """Max over all windows of size ``window`` of the window-mean z-score.

    A contiguous rewrite dilutes a whole-sequence statistic but leaves at least one
    window of intact text; scanning at several scales and keeping the strongest
    window recovers that localised evidence.
    """
    n_seq, L = fused.shape
    if L <= window:
        means = fused.mean(axis=1, keepdims=True)
    else:
        cum = np.cumsum(fused, axis=1)
        window_sums = cum[:, window - 1:].copy()
        window_sums[:, 1:] -= cum[:, : L - window]
        means = window_sums / window
    return means.max(axis=1) / null_std_w


def _union_z(per_token: np.ndarray, cfg: Config, null_var: float) -> np.ndarray:
    """Flagship robust statistic: max(whole-sequence z, multi-scale windowed max z).

    Crucially, normalization uses the **null** per-token variance ``null_var`` (a
    known constant: Var(Gumbel) for the Gumbel channel, gamma*(1-gamma) for the
    green channel), NOT the empirical variance of the (possibly watermarked) data.
    The whole-sequence mean and every window mean are standardized by
    sqrt(null_var / L) or sqrt(null_var / w); the operating threshold is then
    calibrated empirically on the null distribution, which absorbs the
    multiple-testing inflation of taking the max (see pipeline / README).
    """
    n_seq, L = per_token.shape
    whole = per_token.mean(axis=1) / np.sqrt(null_var / L)
    per_scale = [_windowed_max_z(per_token, w, np.sqrt(null_var / w))
                 for w in cfg.window_scales if w < L]
    return np.max(np.vstack([whole] + per_scale), axis=0)


class KGWDetector:
    """Canonical green-list baseline (Kirchenbauer et al., ICML 2023)."""

    name = "kgw_z"

    def score_matrix(self, xs: np.ndarray, hasher: KeyedHasher, cfg: Config) -> np.ndarray:
        ind = _green_per_token(xs, hasher, cfg)
        n_seq, L = xs.shape
        g = ind.sum(axis=1)
        denom = np.sqrt(L * cfg.gamma * (1.0 - cfg.gamma))
        return (g - L * cfg.gamma) / denom

    def detect(self, seq: TokenSeq, hasher: KeyedHasher, cfg: Config) -> DetectionResult:
        xs = np.asarray(seq.tokens, dtype=np.int64).reshape(1, -1)
        z = float(self.score_matrix(xs, hasher, cfg)[0])
        return DetectionResult(score=z, is_watermarked=z > 2.33, scheme=self.name, z=z)


class GumbelDetector:
    """Distortion-free (Aaronson/Greenblatt) scheme, robust multi-scale detection."""

    name = "gumbel_multiscale"

    def score_matrix(self, xs: np.ndarray, hasher: KeyedHasher, cfg: Config) -> np.ndarray:
        sc = _centered_gumbel_per_token(xs, hasher, cfg)
        return _union_z(sc, cfg, _VAR_GUMBEL)

    def detect(self, seq: TokenSeq, hasher: KeyedHasher, cfg: Config) -> DetectionResult:
        xs = np.asarray(seq.tokens, dtype=np.int64).reshape(1, -1)
        z = float(self._whole_z(xs, hasher, cfg)[0])
        return DetectionResult(score=z, is_watermarked=z > 2.33, scheme=self.name, z=z)

    @staticmethod
    def _whole_z(xs, hasher, cfg):
        sc = _centered_gumbel_per_token(xs, hasher, cfg) / np.sqrt(_VAR_GUMBEL)
        return sc.mean(axis=1) / np.sqrt(1.0 / xs.shape[1])


class WmFuseDetector:
    """Flagship: channel-diverse watermark + channel-union multi-scale detection.

    The injected watermark writes signal into *two* independent keyed channels --
    a distortion-free Gumbel tilt (spread over the whole vocabulary) and a green-list
    boost (a ``gamma``-fraction subset). The detector fuses them by taking, per
    channel, the multi-scale union statistic, then the max across channels:

        score = max( gumbel_union_z(x),  kgw_union_z(x) ).

    A single-channel adversary who targets one channel (green_targeted or
    gumbel_targeted) zeros that channel's evidence, but the other channel survives,
    so the union score stays high. Under uniform / contiguous rewrites both channels
    decay together, but their union is at least as powerful as either alone.
    """

    name = "wmfuse_union"

    def score_matrix(self, xs: np.ndarray, hasher: KeyedHasher, cfg: Config) -> np.ndarray:
        gumbel = _centered_gumbel_per_token(xs, hasher, cfg)
        green = _green_per_token(xs, hasher, cfg) - cfg.gamma
        return np.maximum(_union_z(gumbel, cfg, _VAR_GUMBEL),
                          _union_z(green, cfg, cfg.gamma * (1.0 - cfg.gamma)))

    def detect(self, seq: TokenSeq, hasher: KeyedHasher, cfg: Config) -> DetectionResult:
        xs = np.asarray(seq.tokens, dtype=np.int64).reshape(1, -1)
        n_seq, L = xs.shape
        gz = float(GumbelDetector._whole_z(xs, hasher, cfg)[0])
        ind = _green_per_token(xs, hasher, cfg)
        kz = float((ind.sum() - L * cfg.gamma) / np.sqrt(L * cfg.gamma * (1.0 - cfg.gamma)))
        z = max(gz, kz)
        return DetectionResult(score=z, is_watermarked=z > 2.33, scheme=self.name, z=z)


_DETECTORS = {"kgw": KGWDetector, "gumbel": GumbelDetector, "wmfuse": WmFuseDetector}


def get_detector(scheme: str) -> Detector:
    if scheme not in _DETECTORS:
        raise KeyError(f"no detector for scheme {scheme!r}")
    return _DETECTORS[scheme]()
