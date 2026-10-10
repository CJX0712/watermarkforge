# Changelog

## v0.1.0 (2026-10-10)

初始发布（作者：晨星）。

### Added
- 三类水印方案：`kgw`（绿名单）、`gumbel`（失真无关）、`wmfuse`（双通道，消融对照）。
- 多尺度滑动窗口联合检测器（尺度 8/16/32），对连续改写鲁棒。
- 等分位（equal-distortion）解析失真标定：逐 token KL 二分搜索，三类方案同预算公平比较。
- 通道感知攻击：绿名单感知 `green_targeted`、Gumbel 感知 `gumbel_targeted`、连续改写 `contiguous`、均匀 `uniform`。
- 端到端 `cli.py`（`demo` / `verify` / `watermark`）与 `pipeline.WatermarkPipeline`。
- 8 项不变量单测（含确定性逐位一致、绿名单感知攻击负对照）。
- CI workflow（3.12/3.13 lint+pytest+demo 冒烟）；`docs/architecture.md`、`docs/model_card.md`。

### Verification
- 9/9 验收门禁通过（G1–G3 + I1–I6）。
- 确定性：两次同 seed 运行核心指标 `max_abs_diff = 0.0`。
- ruff All checks passed；pytest 8 passed。
- 诚实结论：旗舰 = `gumbel`+联合检测（绿名单感知攻击下胜 KGW）；`wmfuse` 如实记为被否决消融。
