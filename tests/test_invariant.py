"""Determinism, calibration, and null-invariant unit tests.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.seed import set_all
from data.generator import KeyedHasher, SyntheticGenerator
from detect.detectors import KGWDetector, WmFuseDetector, _centered_gumbel_per_token
from eval.benchmark import run_benchmark
from eval.calibrate import calibrate_strength, distortion
from eval.metrics import auc, tpr_at_fpr


def test_determinism_bit_identical():
    cfg = Config()
    r1 = run_benchmark(cfg)
    r2 = run_benchmark(cfg)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        for f in ("auc", "tpr_at_fpr", "mean_kl"):
            assert abs(float(getattr(a, f)) - float(getattr(b, f))) == 0.0


def test_equal_distortion_calibration():
    cfg = Config()
    hasher = KeyedHasher(cfg)
    for s in ("kgw", "gumbel", "wmfuse"):
        st = calibrate_strength(cfg, hasher, s, cfg.target_kl)
        kl = distortion(cfg, hasher, s, st)
        assert abs(kl - cfg.target_kl) <= 0.05 * cfg.target_kl


def test_kgw_null_standard_normal():
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    rng = set_all(7)
    xs = gen.sample_unwatermarked(500, 200, rng)
    z = KGWDetector().score_matrix(xs, hasher, cfg)
    assert abs(np.mean(z)) <= 0.1
    assert 0.9 <= np.std(z) <= 1.1


def test_centered_gumbel_null_mean_zero():
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    rng = set_all(7)
    xs = gen.sample_unwatermarked(200, 100, rng)
    tok = _centered_gumbel_per_token(xs, hasher, cfg)
    assert abs(np.mean(tok)) < 0.05


def test_wmfuse_fpr_control_heldout():
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    rng = set_all(7)
    xs1 = gen.sample_unwatermarked(400, 200, rng)
    xs2 = gen.sample_unwatermarked(400, 200, set_all(8))
    null1 = WmFuseDetector().score_matrix(xs1, hasher, cfg)
    null2 = WmFuseDetector().score_matrix(xs2, hasher, cfg)
    fpr = tpr_at_fpr(null2, null1, cfg.fpr_target)
    assert 0.002 <= fpr <= 0.02


def test_attack_shape_preserved():
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    rng = set_all(7)
    xs = gen.watermark("wmfuse", 20, 100, rng)
    for name in ("apply_random_attack", "apply_contiguous_attack",
                 "apply_green_targeted_attack", "apply_gumbel_targeted_attack"):
        fn = getattr(gen, name)
        out = fn(xs, 0.5, set_all(9))
        assert out.shape == xs.shape
        assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_kgw_collapses_under_green_targeted():
    """KGW's only channel is green; a green-targeted attack must crush it.

    This is the negative-control that motivates the channel-diverse flagship.
    """
    cfg = Config()
    hasher = KeyedHasher(cfg)
    gen = SyntheticGenerator(cfg, hasher)
    for s, v in {"kgw": cfg.kgw_delta, "gumbel": cfg.gumbel_beta, "wmfuse": 1.0}.items():
        gen.set_strength(s, v)
    rng = set_all(7)
    L = 200
    xs_un = gen.sample_unwatermarked(100, L, rng)
    null = KGWDetector().score_matrix(xs_un, hasher, cfg)
    xs_wm = gen.watermark("kgw", 100, L, rng)
    clean = KGWDetector().score_matrix(xs_wm, hasher, cfg)
    att = KGWDetector().score_matrix(gen.apply_green_targeted_attack(xs_wm, 0.7, rng), hasher, cfg)
    clean_tpr = tpr_at_fpr(clean, null, cfg.fpr_target)
    att_tpr = tpr_at_fpr(att, null, cfg.fpr_target)
    assert clean_tpr >= 0.99
    assert att_tpr <= clean_tpr - 0.2


def test_metrics_edge_cases():
    pos = np.array([1.0, 2.0, 3.0])
    neg = np.array([0.0, 0.5, 0.2])
    assert abs(auc(pos, neg) - 1.0) < 1e-9
    # pure separation => TPR 1 at any FPR
    assert tpr_at_fpr(pos, neg, 0.01) == 1.0
