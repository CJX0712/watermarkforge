"""Quickstart: use the WatermarkForge API in 15 lines.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import TokenSeq
from data.generator import KeyedHasher, SyntheticGenerator
from detect.detectors import get_detector
from eval.benchmark import calibrated_strengths
from watermark.schemes import get_scheme


def main() -> None:
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    for s, v in calibrated_strengths(cfg, hasher).items():
        gen.set_strength(s, v)

    rng = set_all(7)
    xs = gen.watermark("gumbel", 1, 64, rng)          # flagship scheme
    seq = TokenSeq(tokens=[int(t) for t in xs[0]])

    scheme = get_scheme("gumbel", cfg, hasher, gen)
    res = scheme.watermark(seq, rng)                  # WatermarkResult
    det = get_detector("gumbel")
    dr = det.detect(seq, hasher, cfg)                 # DetectionResult

    print("watermarked tokens      :", res.tokens)
    print("per-token distortion KL :", round(res.kl, 4))
    print("detection score (z)     :", round(dr.score, 3))
    print("is watermarked?         :", dr.is_watermarked)

    # The same sequence, after a 40% uniform rewrite, scores lower:
    attacked = gen.apply_random_attack(xs, 0.4, set_all(8))
    attacked_score = det.detect(TokenSeq(tokens=[int(t) for t in attacked[0]]), hasher, cfg).score
    print("after 40% rewrite, z    :", round(attacked_score, 3))


if __name__ == "__main__":
    main()
