# WatermarkForge — Architecture

> 作者：晨星 · 纯 NumPy + 标准库，零重型依赖，完全离线可跑。

## 1. 设计目标

在**固定每 token 失真预算**（equal-distortion）下，比较三类 LLM 文本水印方案对**通道感知攻击**（攻击者知道 green-list 或 Gumbel 密钥）的鲁棒性：

- `kgw` — KGW 绿名单水印（Kirchenbauer et al., ICML 2023）
- `gumbel` — 失真无关（distortion-free）Gumbel 水印（Aaronson/Greenblatt; Liu et al., 2024）
- `wmfuse` — 双通道融合（Gumbel tilt + 绿名单 boost），作为**消融对照**

旗舰（flagship）结论见 `README.md`：在同等失真下，**单通道 Gumbel + 多尺度滑动窗口联合检测**在绿名单感知攻击下全面胜出 KGW；双通道 `wmfuse` 因信号稀释而劣于单通道，如实记录为被否决的消融项。

## 2. 模块依赖（单向无环）

```
cli.py ─► pipeline/pipeline.py ─► {data, watermark, detect, eval} ─► core
```

| 模块 | 职责 |
|------|------|
| `core/` | `types`(dataclass) · `errors`(E100~E500) · `config`(schema 校验) · `interfaces`(Protocol) · `seed`(全局确定性 `set_all`) |
| `data/` | 合成 token 序列生成（固定 seed 可复现）+ 通道感知攻击注入 |
| `watermark/` | 三类方案的「单序列」加注/打分 API（`KGWScheme`/`GumbelScheme`/`WmFuseScheme`） |
| `detect/` | 每 token 打分 → 多尺度滑动窗口 z 分数 → 联合检测 |
| `eval/` | `calibrate`(等分位失真标定) · `benchmark`(多 seed 评测) · `metrics`(AUC/TPR@FPR/KL) |
| `pipeline/` | `WatermarkPipeline.run()`：null 不变量 → 头条指标 → 9 项验收门禁 |
| `tests/` | 8 项不变量单测（含确定性逐位一致、绿名单感知攻击负对照） |

## 3. 核心数学

### 3.1 KGW 绿名单
按上下文 hash 选取比例 `γ` 的 green token，对 green logit 加常数 `δ`：
`ℓ̃_x = ℓ_x + δ·𝟙[x∈G(c)]`。检测：整段 green 占比 z 分数 `z = (n_green − γ·L)/√(γ(1−γ)L)`。

### 3.2 失真无关 Gumbel
对每 token 加密钥 Gumbel 噪声：`p(x) ∝ exp((ℓ_x + g_x)/β)`，其中 `g_x = −β·log(−log(u_x))`，`u_x` 由 `(prev,x)` 密钥流式生成。采样分布与 base 分布逐 token KL = `β²π²/6`，与上下文无关 → **失真可解析标定**，无需采样近似。

### 3.3 双通道 wmfuse（消融）
同时注入 Gumbel tilt（强度 `β_w`）与 green boost（强度 `δ_w`），预算在两者间平分。结论：平分稀释两通道，被单通道方案压制。

### 3.4 多尺度滑动窗口联合检测
对每 token 分数序列，在尺度 `s∈{8,16,32}` 上取窗口最大 z 分数，再与整段 z 分数取 `max`：
`z_union = max(z_whole, max_s max_window_s z)`。
- **标准化用 null 方差**（非加注样本经验方差），保证 null 下为标准正态；
- 对**连续改写**（contiguous rewrite）鲁棒：只要任一窗口残留信号即可检出。

### 3.5 等分位失真标定
解析每 token KL（Gumbel 为 `β²π²/6`，KGW 为闭式），对 `β`/`δ` 二分搜索使 `KL = target_kl`（默认 0.3 nats/token），保证三类方案在**同一失真预算**下公平比较。

## 4. 确定性
唯一入口 `core.seed.set_all(seed)` 一次设齐 numpy/random；`gumbel` 通道均值按每上下文居中，使 null 每 token 分数均值恰为 0。两次同 seed 运行 `benchmark.json` 核心指标**逐位一致**（实测 `max_abs_diff = 0.0`）。
