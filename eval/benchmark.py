"""Benchmark runner: equal-distortion, multi-seed evaluation of schemes x attacks.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import BenchmarkRecord
from data.generator import KeyedHasher, SyntheticGenerator
from detect.detectors import get_detector
from eval.calibrate import calibrate_strength, distortion
from eval.metrics import auc, tpr_at_fpr

SCHEMES = ("kgw", "gumbel", "wmfuse")


def _apply_attack(gen: SyntheticGenerator, xs: np.ndarray, attack: str,
                  rate: float, rng: np.random.Generator) -> np.ndarray:
    if attack == "uniform":
        return gen.apply_random_attack(xs, rate, rng)
    if attack == "contiguous":
        return gen.apply_contiguous_attack(xs, rate, rng)
    if attack == "green_targeted":
        return gen.apply_green_targeted_attack(xs, rate, rng)
    if attack == "gumbel_targeted":
        return gen.apply_gumbel_targeted_attack(xs, rate, rng)
    raise ValueError(f"unknown attack {attack!r}")


def calibrated_strengths(cfg: Config, hasher: KeyedHasher) -> Dict[str, float]:
    """Map each scheme -> strength scalar that equalizes its analytic distortion.

    With ``equal_distortion`` off we fall back to the raw config strengths, so the
    runner can be used for either a fairness-controlled or a fixed-strength study.
    """
    if not cfg.equal_distortion:
        return {"kgw": cfg.kgw_delta, "gumbel": cfg.gumbel_beta, "wmfuse": 1.0}
    return {s: calibrate_strength(cfg, hasher, s, cfg.target_kl) for s in SCHEMES}


def run_benchmark(cfg: Config) -> List[BenchmarkRecord]:
    """One record per (seed, length, scheme, attack, rate). Fully deterministic."""
    records: List[BenchmarkRecord] = []
    for seed in cfg.seeds:
        rng = set_all(seed)
        hasher = KeyedHasher(cfg)
        strengths = calibrated_strengths(cfg, hasher)
        gen = SyntheticGenerator(cfg, hasher)
        for s, v in strengths.items():
            gen.set_strength(s, v)
        for length in cfg.lengths:
            xs_un = gen.sample_unwatermarked(cfg.n_seq, length, rng)
            for scheme in SCHEMES:
                det = get_detector(scheme)
                xs_wm = gen.watermark(scheme, cfg.n_seq, length, rng)
                null = det.score_matrix(xs_un, hasher, cfg)
                clean = det.score_matrix(xs_wm, hasher, cfg)
                kl = distortion(cfg, hasher, scheme, strengths[scheme])
                records.append(BenchmarkRecord(
                    scheme=scheme, detector=det.name, length=length, attack="none",
                    attack_rate=0.0, auc=auc(clean, null),
                    tpr_at_fpr=tpr_at_fpr(clean, null, cfg.fpr_target),
                    mean_kl=kl, n_seq=cfg.n_seq, seed=seed,
                ))
                for attack in cfg.attacks:
                    for rate in cfg.attack_rates:
                        xs_att = _apply_attack(gen, xs_wm, attack, rate, rng)
                        att = det.score_matrix(xs_att, hasher, cfg)
                        records.append(BenchmarkRecord(
                            scheme=scheme, detector=det.name, length=length, attack=attack,
                            attack_rate=rate, auc=auc(att, null),
                            tpr_at_fpr=tpr_at_fpr(att, null, cfg.fpr_target),
                            mean_kl=kl, n_seq=cfg.n_seq, seed=seed,
                        ))
    return records


def _pick(records: List[BenchmarkRecord], scheme: str, length: int,
          attack: str, rate: float) -> List[BenchmarkRecord]:
    return [r for r in records if r.scheme == scheme and r.length == length
            and r.attack == attack and r.attack_rate == rate]


def aggregate(records: List[BenchmarkRecord]) -> dict:
    """mean +/- std over seeds, keyed by '<scheme>|L<len>|<attack>|<rate>'."""
    out: dict = {}
    lengths = sorted({r.length for r in records})
    cells = sorted({(r.attack, r.attack_rate) for r in records})
    for scheme in SCHEMES:
        for length in lengths:
            for attack, rate in cells:
                sel = _pick(records, scheme, length, attack, rate)
                if not sel:
                    continue
                key = f"{scheme}|L{length}|{attack}|{rate}"
                out[key] = {
                    "scheme": scheme,
                    "length": length,
                    "attack": attack,
                    "attack_rate": rate,
                    "auc_mean": float(np.mean([r.auc for r in sel])),
                    "auc_std": float(np.std([r.auc for r in sel])),
                    "tpr_mean": float(np.mean([r.tpr_at_fpr for r in sel])),
                    "tpr_std": float(np.std([r.tpr_at_fpr for r in sel])),
                    "kl_mean": float(np.mean([r.mean_kl for r in sel])),
                    "n_seeds": len(sel),
                }
    return out


def cfg_lengths(records: List[BenchmarkRecord]) -> List[int]:
    return sorted({r.length for r in records})


def cfg_cells(records: List[BenchmarkRecord]) -> List[tuple]:
    cells = {(r.attack, r.attack_rate) for r in records}
    order = {"none": 0}
    return sorted(cells, key=lambda c: (order.get(c[0], 1), c[0], c[1]))
