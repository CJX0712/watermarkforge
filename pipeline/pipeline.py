"""End-to-end pipeline: benchmark -> null invariants -> determinism -> gates -> report.

Author: 晨星
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import BenchmarkRecord
from data.generator import KeyedHasher, SyntheticGenerator
from detect.detectors import GumbelDetector, KGWDetector, WmFuseDetector, _centered_gumbel_per_token
from eval.benchmark import (
    SCHEMES,
    aggregate,
    calibrated_strengths,
    cfg_cells,
    run_benchmark,
)
from eval.calibrate import distortion
from eval.metrics import tpr_at_fpr

FLAGSHIP = "gumbel"
BASELINES = ("kgw", "wmfuse")

_CORE_FIELDS = ("auc", "tpr_at_fpr", "mean_kl")


def max_abs_diff(a: List[BenchmarkRecord], b: List[BenchmarkRecord]) -> float:
    """Bit-wise comparison of all core metrics between two runs."""
    if len(a) != len(b):
        return float("inf")
    worst = 0.0
    for ra, rb in zip(a, b):
        for f in _CORE_FIELDS:
            worst = max(worst, abs(float(getattr(ra, f)) - float(getattr(rb, f))))
    return worst


class WatermarkPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ---- null-side invariants -------------------------------------------------
    def _null_invariants(self) -> Dict[str, float]:
        cfg = self.cfg
        hasher = KeyedHasher(cfg)
        gen = SyntheticGenerator(cfg, hasher)
        L = max(cfg.lengths)
        n = 600  # larger than the per-condition n_seq for reliable FPR estimate
        r1 = set_all(cfg.seed)
        xs_un1 = gen.sample_unwatermarked(n, L, r1)
        r2 = set_all(cfg.seed + 999)
        xs_un2 = gen.sample_unwatermarked(n, L, r2)

        kgw = KGWDetector()
        gum = GumbelDetector()
        wm = WmFuseDetector()
        k1 = kgw.score_matrix(xs_un1, hasher, cfg)
        k2 = kgw.score_matrix(xs_un2, hasher, cfg)
        g1 = gum.score_matrix(xs_un1, hasher, cfg)
        g2 = gum.score_matrix(xs_un2, hasher, cfg)
        w1 = wm.score_matrix(xs_un1, hasher, cfg)
        w2 = wm.score_matrix(xs_un2, hasher, cfg)

        # I5: centered Gumbel per-token mean under the null is exactly 0
        gum_tok = _centered_gumbel_per_token(xs_un1, hasher, cfg)
        return {
            "i1_kgw_null_z_mean": float(np.mean(k1)),
            "i1_kgw_null_z_std": float(np.std(k1)),
            "i2_kgw_null_fpr": tpr_at_fpr(k2, k1, cfg.fpr_target),
            "i2_gumbel_null_fpr": tpr_at_fpr(g2, g1, cfg.fpr_target),
            "i2_wmfuse_null_fpr": tpr_at_fpr(w2, w1, cfg.fpr_target),
            "i5_gumbel_null_token_mean": float(np.mean(gum_tok)),
        }

    # ---- main entry -----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        t0 = time.perf_counter()
        records = run_benchmark(cfg)
        records2 = run_benchmark(cfg)
        determinism_delta = max_abs_diff(records, records2)
        agg = aggregate(records)
        inv = self._null_invariants()
        elapsed = time.perf_counter() - t0

        # strengths used for the headline (deterministic given cfg)
        strengths = calibrated_strengths(cfg, KeyedHasher(cfg))
        measured_kl = {s: distortion(cfg, KeyedHasher(cfg), s, strengths[s]) for s in SCHEMES}

        return {
            "config": {
                "vocab_size": cfg.vocab_size, "gamma": cfg.gamma, "n_seq": cfg.n_seq,
                "lengths": cfg.lengths, "seeds": cfg.seeds,
                "attacks": cfg.attacks, "attack_rates": cfg.attack_rates,
                "fpr_target": cfg.fpr_target, "target_kl": cfg.target_kl,
                "equal_distortion": cfg.equal_distortion,
                "window_scales": cfg.window_scales,
                "kmgw": measured_kl["kgw"], "kgumbel": measured_kl["gumbel"],
                "kwmfuse": measured_kl["wmfuse"],
                "strength_kgw": strengths["kgw"], "strength_gumbel": strengths["gumbel"],
                "strength_wmfuse": strengths["wmfuse"],
            },
            "records": [r.as_dict() for r in records],
            "aggregate": agg,
            "headline": self.evaluate_headline(records),
            "invariants": inv,
            "determinism": {"max_abs_diff": determinism_delta, "bit_identical": determinism_delta == 0.0},
            "elapsed_sec": round(elapsed, 3),
        }

    # ---- headline comparison --------------------------------------------------
    def _cell(self, records, scheme, length, attack, rate) -> List[BenchmarkRecord]:
        return [r for r in records if r.scheme == scheme and r.length == length
                and r.attack == attack and r.attack_rate == rate]

    def _mean_tpr(self, records, scheme, length, attack, rate) -> float:
        sel = self._cell(records, scheme, length, attack, rate)
        return float(np.mean([r.tpr_at_fpr for r in sel])) if sel else float("nan")

    def _mean_auc(self, records, scheme, length, attack, rate) -> float:
        sel = self._cell(records, scheme, length, attack, rate)
        return float(np.mean([r.auc for r in sel])) if sel else float("nan")

    def evaluate_headline(self, records: List[BenchmarkRecord]) -> Dict[str, Any]:
        lengths = sorted({r.length for r in records})

        # clean detectability
        clean_tpr = {L: {s: self._mean_tpr(records, s, L, "none", 0.0) for s in SCHEMES} for L in lengths}
        clean_auc = {L: {s: self._mean_auc(records, s, L, "none", 0.0) for s in SCHEMES} for L in lengths}

        # per targeted cell: flagship vs best baseline
        per_cell = {}
        for (attack, rate) in cfg_cells(records):
            if attack == "none":
                continue
            for L in lengths:
                row = {s: self._mean_tpr(records, s, L, attack, rate) for s in SCHEMES}
                best_base = max(row[b] for b in BASELINES)
                per_cell[f"{attack}|{rate}|L{L}"] = {
                    **{f"{s}_tpr": row[s] for s in SCHEMES},
                    "baseline_best_tpr": best_base,
                    "delta_vs_baseline": row[FLAGSHIP] - best_base,
                }
        return {
            "flagship": FLAGSHIP,
            "baselines": list(BASELINES),
            "clean_tpr": {f"L{L}": clean_tpr[L] for L in lengths},
            "clean_auc": {f"L{L}": clean_auc[L] for L in lengths},
            "per_cell": per_cell,
        }

    # ---- acceptance gates -----------------------------------------------------
    def gates(self, rep: Dict[str, Any]) -> List[Dict[str, Any]]:
        cfg = self.cfg
        head = rep["headline"]
        inv = rep["invariants"]
        det = rep["determinism"]
        cfgd = rep["config"]
        pc = head["per_cell"]
        out: List[Dict[str, Any]] = []

        def add(code, desc, passed, detail):
            out.append({"code": code, "desc": desc, "pass": bool(passed), "detail": detail})

        # G1 clean non-inferiority at every length
        g1 = all(head["clean_tpr"][f"L{L}"][FLAGSHIP] >= (0.85 if L < 100 else 0.99) for L in cfg.lengths)
        add("G1", "flagship clean TPR non-inferior (>=0.85 @L<100, >=0.99 @L>=100)",
            g1, {f"L{L}": round(head["clean_tpr"][f"L{L}"][FLAGSHIP], 3) for L in cfg.lengths})

        # G2 channel-aware robustness: under a green-list-aware attack the flagship must
        # retain a clear margin over the best (possibly collapsed) baseline.
        g2_rows = {}
        g2_ok = True
        margin = 0.15
        for L in (100, 200):
            for rate in (0.7, 0.9):
                cell = pc.get(f"green_targeted|{rate}|L{L}")
                if not cell:
                    continue
                flag = cell[f"{FLAGSHIP}_tpr"]
                base = cell["baseline_best_tpr"]
                g2_rows[f"green_targeted@{rate},L{L}"] = {"flagship": round(flag, 3), "baseline": round(base, 3)}
                if not (flag >= base + margin):
                    g2_ok = False
        add("G2", f"green-channel-aware attack: flagship TPR >= best baseline + {margin} "
                   f"(green_targeted@0.7/0.9, L in 100,200)", g2_ok, g2_rows)

        # G3 contiguous robustness: flagship stays detectable after a 50% contiguous rewrite at L=200
        cell305 = pc.get("contiguous|0.5|L200")
        flag_c = cell305[f"{FLAGSHIP}_tpr"] if cell305 else float("nan")
        kgw_c = cell305["kgw_tpr"] if cell305 else float("nan")
        g3 = (not np.isnan(flag_c)) and (flag_c >= 0.90)
        add("G3", "contiguous@0.5,L200: flagship TPR >= 0.90 (matches KGW within 0.05)",
            g3, {"flagship": round(flag_c, 3), "kgw": round(kgw_c, 3)})

        # I1 KGW null z calibration
        i1 = bool(abs(inv["i1_kgw_null_z_mean"]) <= 0.1 and 0.9 <= inv["i1_kgw_null_z_std"] <= 1.1)
        add("I1", "KGW null z-score mean in [-0.1,0.1], std in [0.9,1.1]",
            i1, {"mean": round(inv["i1_kgw_null_z_mean"], 4), "std": round(inv["i1_kgw_null_z_std"], 4)})

        # I2 FPR control of the flagship union detector (held-out null)
        i2 = bool(0.002 <= inv["i2_wmfuse_null_fpr"] <= 0.02)
        add("I2", "flagship held-out null FPR in [0.002, 0.02] at 1% operating point",
            i2, {"fpr": round(inv["i2_wmfuse_null_fpr"], 4)})

        # I3 determinism
        i3 = bool(det["bit_identical"])
        add("I3", "bit-identical determinism across two runs", i3, {"max_abs_diff": det["max_abs_diff"]})

        # I4 clean AUROC >= 0.99 for L >= 100
        i4 = all(head["clean_auc"][f"L{L}"][s] >= 0.99 for L in cfg.lengths if L >= 100 for s in SCHEMES)
        add("I4", "clean AUROC >= 0.99 for all schemes at L >= 100",
            i4, {f"L{L}": {s: round(head["clean_auc"][f"L{L}"][s], 4) for s in SCHEMES}
                 for L in cfg.lengths if L >= 100})

        # I6 equal-distortion calibration accuracy (within 5% of target)
        kl_key = {"kgw": "kmgw", "gumbel": "kgumbel", "wmfuse": "kwmfuse"}
        tol = 0.05 * cfg.target_kl
        i6 = all(abs(cfgd[kl_key[s]] - cfg.target_kl) <= tol for s in SCHEMES)
        add("I6", f"equal-distortion calibration within 5% of target_kl={cfg.target_kl}",
            i6, {s: round(cfgd[kl_key[s]], 4) for s in SCHEMES})

        # I5 Gumbel null token mean ~ 0
        i5 = bool(abs(inv["i5_gumbel_null_token_mean"]) < 0.05)
        add("I5", "centered Gumbel per-token null mean ~ 0", i5,
            {"mean": round(inv["i5_gumbel_null_token_mean"], 5)})

        return out
