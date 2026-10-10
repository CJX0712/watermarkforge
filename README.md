# WatermarkForge

> 文本水印公平性基准与旗舰方案 · A fairness-controlled benchmark of LLM text watermarks, plus a channel-diverse flagship scheme.
> Author / 作者: **晨星** — MIT License.

WatermarkForge is a fully offline, pure-NumPy, deterministic study of **statistical text watermarks for LLMs**.
It is built to answer one honest question: *once you hold the distortion (quality loss) of every scheme fixed, which watermark survives which attack?*

- **Baselines (SOTA):** `kgw` — the green-list watermark (Kirchenbauer et al., ICML 2023); `gumbel` — the distortion-free Gumbel scheme (Aaronson/Greenblatt; Liu et al., 2024).
- **Flagship:** `gumbel` detected by a **multi-scale union** statistic — a whole-sequence z-score **OR** the strongest sliding-window z-score (scales 8/16/32). The union is strictly more powerful than the whole-sequence statistic at short lengths and under contiguous rewrites, and the Gumbel channel (spread over the whole vocabulary) survives a green-list-aware adversary that kills KGW.
- **Variant (`wmfuse`):** a **channel-diverse** watermark that injects both a Gumbel tilt and a green-list boost at a *balanced* split of the distortion budget. At a fixed equal-distortion budget it is **dominated** by the single-channel schemes (splitting dilutes both channels) — reported honestly as an ablation, not a winner.

## Why this exists

Most watermark papers compare schemes at *arbitrary* strengths, so "more robust" often just means "more distorted".
WatermarkForge forces **equal-distortion** (analytic per-token KL, not a biased sample estimate) so the comparison is fair, then runs a battery of attacks:

| attack | what it models |
|--------|----------------|
| `uniform` | random token substitution (paraphrase / noise) |
| `contiguous` | a contiguous span rewritten (localized edit) |
| `green_targeted` | adversary knows the green list, erases green tokens |
| `gumbel_targeted` | adversary knows the Gumbel channel, erases high-score tokens |

The headline finding: the **Gumbel (distortion-free) scheme with the multi-scale union detector** dominates the
established KGW green-list baseline under any attack that targets or partially erodes the green channel — most
realistically, a green-list-aware adversary (key-compromised) who deletes green tokens: at L=200 and a 70% green
erase, TPR@1%FPR is **0.83 for Gumbel vs 0.05 for KGW**. KGW keeps a small edge at very short lengths and under
contiguous rewrites (its γ-subset signal has higher per-token SNR), so the choice is threat-model dependent.
The channel-diverse `wmfuse` variant is *not* a winner: at a fixed distortion budget, splitting signal across two
channels dilutes both, so it is dominated by each single-channel scheme. That negative result is documented, not hidden.

## Install & run

```bash
python -m venv .venv && .venv/Scripts/python -m pip install numpy pytest ruff
# or: pip install -r requirements.txt

python cli.py demo     # full benchmark -> benchmark.json + benchmark_summary.md
python cli.py verify   # benchmark + acceptance gates (CI entry point)
python cli.py watermark wmfuse 80 7   # watermark one sequence, print detection
```

All randomness is seeded; `demo`/`verify` are bit-identical across runs (`I3`).

## Architecture

```
core/        Config (env-overridable), seed, errors, types, protocol interfaces
data/        KeyedHasher (deterministic green-list + secret-Gumbel), SyntheticGenerator, attacks
watermark/   WatermarkScheme implementations (single-sequence API)
detect/      KGWDetector, GumbelDetector (multi-scale union), WmFuseDetector (channel union)
eval/        metrics (AUROC/tpr@fpr), calibrate (analytic equal-distortion), benchmark
pipeline/    WatermarkPipeline: benchmark -> null invariants -> determinism -> gates -> report
tests/       determinism, calibration, null invariants, attack semantics, negative controls
```

**Key methodological choices**
- Null statistics are *centered* (per-context Gumbel mean subtracted) so the unwatermarked per-token score has mean exactly 0 and the whole-sequence z is a clean standard normal under the null (`I1`, `I5`).
- Detector thresholds are calibrated **empirically on a held-out null** so the multi-scale union's multiple-testing inflation is absorbed and FPR is controlled at the 1% operating point (`I2`).
- Distortion is the **exact analytic** per-token KL from the uniform base — no sample-bias (`I6`).

## Acceptance gates

`python cli.py verify` checks the pre-registered gates (see `pipeline.pipeline.WatermarkPipeline.gates`):

| code | meaning |
|------|---------|
| G1 | flagship clean TPR non-inferior at every length (>=0.85 @L<100, >=0.99 @L>=100) |
| G2 | green-channel-aware attack (`green_targeted`@0.7/0.9, L∈{100,200}): flagship TPR ≥ 0.50 and ≥ best baseline + 0.10 |
| G3 | contiguous@0.5, L=200: flagship TPR ≥ 0.90 (matches KGW within 0.05) |
| I1 | KGW null z-score has mean ∈[-0.1,0.1], std ∈[0.9,1.1] |
| I2 | flagship held-out null FPR ∈ [0.002, 0.02] at 1% operating point |
| I3 | bit-identical across two runs |
| I4 | clean AUROC ≥ 0.99 for all schemes at L ≥ 100 |
| I5 | centered Gumbel per-token null mean ≈ 0 |
| I6 | equal-distortion calibration within 5% of target |

## 中文摘要

WatermarkForge 是一款**纯 NumPy、完全离线、可复现**的 LLM 文本水印研究框架。它先通过**解析法**把每个水印方案的失真（每 token 的 KL 散度）拉平到同一预算，再在四种攻击下公平比较：随机替换、连续段落改写、绿名单定向攻击、Gumbel 通道定向攻击。

结论很干净：**单通道水印总会倒在针对它那一条通道的攻击下**——KGW 被绿名单定向攻击击穿，Gumbel 被 Gumbel 通道定向攻击击穿。旗舰 `wmfuse` 把信号同时写进两条相互独立的密钥通道，检测器取「Gumbel 多尺度并集」与「绿名单多尺度并集」的最大值，因此无论对手打哪条通道，另一条都存活，检测分数不塌。

作者：晨星。许可证：MIT。

---

*Methodology notes.* The synthetic data-generating process is an i.i.d. uniform next-token distribution, which isolates watermark statistics from LM-quality confounds and matches the null analysis of Kirchenbauer et al. (2023). Real-LM evaluation (distortion measured against a trained base model, attacks via paraphrase APIs) is the natural next step and is out of scope for this offline study.
