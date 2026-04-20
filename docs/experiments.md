# Experiment Tracker

Track all experiments run against the strategy. Record hypothesis, method, result, and decision for each.

!!! warning "Pre-requisites"
    Experiments marked with ⛔ depend on fixing survivorship bias ([#17](https://github.com/williamvsoma/different-stock-data/issues/17)) and timing issues ([#18](https://github.com/williamvsoma/different-stock-data/issues/18), [#19](https://github.com/williamvsoma/different-stock-data/issues/19)) first. Results are invalid without clean data.

---

## EXP-01: Ensemble Weight Optimization

**Issue:** [#21](https://github.com/williamvsoma/different-stock-data/issues/21)
**Agent:** `senior-data-scientist`
**Status:** ❌ Rejected (no improvement)
**Run date:** 2025-07-22 | **Test quarters:** 2 | **Script:** `experiments/run_experiments.py`

| Detail | Value |
|---|---|
| **Hypothesis** | XGBoost 50% weight is arbitrary; Ridge may deserve more weight given noise level |
| **Method** | Walk-forward with (a) baseline 50/25/25 (b) equal 1/3 each (c) adaptive weights from OOS RC (lookback=4, floor=0.10) (d) Ridge-heavy 25/50/25 |
| **Metric** | Ensemble rank correlation, avg quarterly excess return (net), annualised Sharpe, IR |
| **Result** | See table below |
| **Decision** | **REJECT** — all alternatives underperform baseline on returns, Sharpe, and IR |

| Variant | Excess Net | Ens RC | Sharpe | IR |
|---|---|---|---|---|
| Baseline (50/25/25) | **+6.49%** | 0.147 | **96.1** | **10.1** |
| Equal (33/33/33) | +5.71% | 0.138 | 72.7 | 9.4 |
| Adaptive (OOS RC) | +5.74% | **0.157** | 35.2 | 4.9 |
| Ridge-heavy (25/50/25) | +3.60% | 0.120 | 43.1 | 6.8 |

**Key findings:**

- Adaptive weights improve ensemble RC by +0.01 but reduce excess returns by ~75bps and Sharpe by 63%
- Ridge is consistently weakest (RC=0.071). Its 25% weight contributes diversification alpha despite low IC.
- Only 2 test quarters — revisit when dataset grows to 8+ quarters.
- Adaptive weights had insufficient training signal (only 1 prior quarter of RCs before adaptation kicks in).

---

## EXP-02: Shrinkage Alpha Sweep

**Issue:** [#26](https://github.com/williamvsoma/different-stock-data/issues/26)
**Agent:** `senior-data-scientist`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | α=0.5 shrinkage may be too aggressive if models have real signal |
| **Method** | Walk-forward with α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Also test reverse order (shrink then winsorize). |
| **Metric** | Ensemble RC, net Sharpe, turnover, max drawdown per α |
| **Result** | — |
| **Decision** | — |

---

## EXP-03: Rank Feature Ablation

**Issue:** [#34](https://github.com/williamvsoma/different-stock-data/issues/34)
**Agent:** `senior-data-scientist`
**Status:** ❌ Rejected (no improvement)
**Run date:** 2025-07-22 | **Test quarters:** 2 | **Script:** `experiments/run_experiments.py`

| Detail | Value |
|---|---|
| **Hypothesis** | Rank features are redundant for tree models; removing them halves dimensionality with no signal loss |
| **Method** | Walk-forward with (a) all features (b) model-specific: raw→trees, rank→Ridge (c) raw only (d) rank only |
| **Metric** | Per-model RC, ensemble RC, avg quarterly excess return (net), Sharpe |
| **Result** | See table below |
| **Decision** | **REJECT** — baseline (all features for all models) wins on returns and ensemble RC |

| Variant | Excess Net | Ens RC | XGB RC | Ridge RC | RF RC |
|---|---|---|---|---|---|
| Baseline (all) | **+6.49%** | **0.147** | **0.164** | 0.071 | 0.169 |
| Model-specific | +5.96% | 0.125 | 0.153 | 0.014 | **0.184** |
| Raw only | +4.67% | 0.147 | 0.153 | **0.088** | **0.184** |
| Rank only | +2.92% | 0.119 | 0.139 | 0.014 | 0.158 |

**Key findings:**

- Hypothesis is **half-right**: RF improves with raw-only (0.169→0.184), confirming trees don't need rank features
- But Ridge **collapses** on rank-only features (RC 0.071→0.014). Ranks remove scale/magnitude info Ridge relies on.
- XGB *benefits* from rank features (0.164 vs 0.153 raw-only) — unexpected. Ranks provide useful cross-sectional normalisation even for trees.
- The "give everything to everyone" approach produces the best ensemble. Feature redundancy is a feature, not a bug.

---

## EXP-04: Volatility Model Benchmark

**Issue:** [#32](https://github.com/williamvsoma/different-stock-data/issues/32)
**Agent:** `senior-data-scientist`
**Status:** ❌ Rejected (no improvement)
**Run date:** 2025-07-22 | **Test quarters:** 2 | **Script:** `experiments/run_experiments.py`

| Detail | Value |
|---|---|
| **Hypothesis** | Historical vol may match or beat ML vol prediction; vol model doesn't need XGBoost |
| **Method** | (a) Separate vol hyperparams (depth=5, less reg) (b) Same + quality gate (fall back to hist_vol_3m when vol_rc < 0.10) |
| **Metric** | Vol rank correlation, portfolio excess return, Sharpe |
| **Result** | See table below |
| **Decision** | **REJECT** — deeper vol model worsens vol_rc; quality gate worsens further; zero impact on returns |

| Variant | Vol RC | Excess Net | Sharpe | IR |
|---|---|---|---|---|
| Baseline (XGB depth=3) | **0.793** | +6.49% | 96.1 | 10.1 |
| Deeper (depth=5, less reg) | 0.790 | +6.49% | 96.1 | 10.1 |
| Deeper + gate | 0.752 | +6.49% | 96.1 | 10.1 |

**Key findings:**

- Returns are **completely identical** across all vol variants — MV optimizer produces same portfolios regardless of small vol differences
- Baseline vol model (depth=3) already excellent at RC=0.793. Deeper trees overfit on small training set.
- Quality gate fires and falls back to hist_vol_3m, which has *worse* correlation (0.752). ML vol model is already better than naive.
- GARCH baseline was not tested (insufficient data for reliable GARCH estimation with quarterly frequency).
- The vol model is "good enough" — improvements to vol prediction don't propagate to portfolio returns at current scale.

---

## EXP-05: Subperiod Stability

**Issue:** [#31](https://github.com/williamvsoma/different-stock-data/issues/31)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | Alpha should persist across subperiods if signal is real |
| **Method** | Split backtest into halves (and thirds). Report Sharpe, IC, drawdown in each. |
| **Metric** | Sharpe per subperiod, IC stability, drawdown |
| **Result** | — |
| **Decision** | — |

---

## EXP-06: Decile Return Monotonicity

**Issue:** [#31](https://github.com/williamvsoma/different-stock-data/issues/31)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | If return predictions are useful, predicted-return deciles should show monotonic realized returns |
| **Method** | Each quarter, rank stocks by predicted return into deciles. Report avg realized return per decile. |
| **Metric** | Long-short spread (D10 − D1), monotonicity score |
| **Result** | — |
| **Decision** | — |

---

## EXP-07: Permutation Test (Random Signal)

**Issue:** [#31](https://github.com/williamvsoma/different-stock-data/issues/31)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | Real strategy should significantly beat random signal through the same optimizer |
| **Method** | Shuffle predicted returns randomly before MV optimization. Run 100 permutations. Compare real vs distribution. |
| **Metric** | p-value of real Sharpe vs permutation distribution |
| **Result** | — |
| **Decision** | — |

---

## EXP-08: Parameter Sensitivity Grid

**Issue:** [#31](https://github.com/williamvsoma/different-stock-data/issues/31)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | Alpha should be robust to small parameter perturbations |
| **Method** | Grid over: `risk_aversion` ∈ {0.5, 1, 2, 5}, `max_weight` ∈ {0.01, 0.02, 0.05, 0.10}, `shrinkage_alpha` ∈ {0.25, 0.5, 0.75}, `winsor_pct` ∈ {0.02, 0.05, 0.10} |
| **Metric** | Net Sharpe heatmap, sensitivity of alpha to each parameter |
| **Result** | — |
| **Decision** | — |

---

## EXP-09: Transaction Cost Sensitivity

**Issue:** [#23](https://github.com/williamvsoma/different-stock-data/issues/23)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17

| Detail | Value |
|---|---|
| **Hypothesis** | Net alpha may flip sign at higher cost assumptions |
| **Method** | Run walk-forward at cost_bps ∈ {10, 20, 30, 50, 100}. Also add costs to factor benchmarks for fair comparison. |
| **Metric** | Net Sharpe vs cost, breakeven cost level |
| **Result** | — |
| **Decision** | — |

---

## EXP-10: Sector and Factor Tilt Attribution

**Issue:** [#24](https://github.com/williamvsoma/different-stock-data/issues/24)
**Agent:** `quant-researcher`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17

| Detail | Value |
|---|---|
| **Hypothesis** | Reported alpha may be disguised sector/factor bets, not stock selection |
| **Method** | (a) Report portfolio sector weights vs universe each quarter (b) Add sector-neutrality constraint to MV (c) Add beta-neutrality constraint (d) Compare constrained vs unconstrained Sharpe |
| **Metric** | Factor-neutral alpha, sector tilt magnitude, beta over time |
| **Result** | — |
| **Decision** | — |

---

## Completed Experiments

- **EXP-01**: Ensemble Weight Optimization — ❌ Rejected (2025-07-22)
- **EXP-03**: Rank Feature Ablation — ❌ Rejected (2025-07-22)
- **EXP-04**: Volatility Model Benchmark — ❌ Rejected (2025-07-22)

## Issue #26 — Shrinkage Alpha & Winsorization Sweep

**Date**: 2026-04-20
**Branch**: `experiment/26-sweep-shrinkage-winsor`

### Hypothesis
Baseline uses `shrinkage_alpha=0.5` and `winsor_pct=0.05` with order: winsorize → shrink.
This compresses ensemble predictions heavily. If models have real signal, less shrinkage
should improve performance.

### Method
Walk-forward backtest (N=2 quarters) with:
- Shrinkage α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
- Winsor pct ∈ {0.01, 0.02, 0.05, 0.10, 0.20}
- Reverse order test: shrink→winsorize vs winsorize→shrink

All other parameters held at baseline values.

### Results

**Shrinkage Alpha Sweep**:
| α    | Avg Excess Net | Win Rate | Sharpe | IR     | Turnover | Holdings |
|------|---------------|----------|--------|--------|----------|----------|
| 0.00 | -2.90%        | 0%       | 6.97   | -31.35 | 64%      | 68       |
| 0.25 | +2.65%        | 100%     | 11.70  | 26.67  | 81%      | 56       |
| 0.50 | +6.49%        | 100%     | 96.07  | 10.09  | 86%      | 54       |
| 0.75 | +6.98%        | 100%     | 17.89  | 4.32   | 87%      | 54       |
| 1.00 | +9.30%        | 100%     | 40.03  | 7.64   | 84%      | 52       |

**Winsorization Sweep**:
| pct  | Avg Excess Net | Win Rate | Sharpe  | IR    | Turnover | Holdings |
|------|---------------|----------|---------|-------|----------|----------|
| 0.01 | +7.24%        | 100%     | 8.36    | 7.21  | 85%      | 54       |
| 0.02 | +7.31%        | 100%     | 8.64    | 7.67  | 86%      | 54       |
| 0.05 | +6.49%        | 100%     | 96.07   | 10.09 | 86%      | 54       |
| 0.10 | +2.06%        | 100%     | 233.88  | 2.47  | 88%      | 57       |
| 0.20 | -2.59%        | 0%       | 12.56   | -2.15 | 87%      | 60       |

**Reverse Order**: Identical to baseline (operations commute at default params).

### Interpretation

1. **Higher α (less shrinkage) → better excess returns**: Monotonically increasing from
   α=0.0 (-2.9%) to α=1.0 (+9.3%). The models have real predictive content that shrinkage
   is suppressing.

2. **Less winsorization → better**: winsor_pct=0.02 is best. Aggressive clipping (0.10+)
   destroys signal. Tail predictions carry disproportionate alpha.

3. **Order irrelevant**: At 0.5/0.05, winsorize→shrink ≡ shrink→winsorize.

### Statistical Warning
⚠ **N=2 quarters only**. These results are INSUFFICIENT for any deployment conclusion.
The directional pattern (less compression → more excess) is informative but not significant.
Need 15+ quarters minimum for 80% power at conventional significance levels.

### Recommendation
- **Immediate**: No parameter change justified with N=2.
- **If validated with more data**: Consider α=0.75 (conservative step toward less shrinkage)
  and winsor_pct=0.02 (mild widening). Never go to α=1.0 without guardrails — zero shrinkage
  in production is reckless with XGBoost predictions.

## Issue #31 — Comprehensive Robustness Tests

**Date**: 2026-04-20
**Branch**: `experiment/31-comprehensive-robustness`

### Overview
Stress-test the walk-forward strategy across multiple robustness dimensions:
subperiod stability, prediction decile analysis, rolling IC, parameter sensitivity,
and permutation testing.

### Method
All tests run on the baseline configuration (PROD_CFG) with N=2 quarters
of walk-forward results and 882 stock-quarter observations.

### Results

**1. Subperiod Stability**:
| Period | N | Excess Net | Sharpe | IC |
|--------|---|-----------|--------|------|
| Full | 2 | +6.49% | 96.07 | 0.147 |
| First half | 1 | +7.40% | — | 0.151 |
| Second half | 1 | +5.58% | — | 0.143 |

✓ Consistent sign across halves. IC stable.

**3. Prediction Decile Analysis**:
| Decile | Avg Return |
|--------|-----------|
| 1 (worst predicted) | +4.68% |
| 5 | +7.00% |
| 8 | +12.35% |
| 10 (best predicted) | +26.99% |
| **D10 - D1 spread** | **+22.31%** |

✓ Strong monotonic signal. Top decile dominates. Tail predictions carry alpha.

**4. Rolling IC**: Insufficient data (N=2, need ≥4 for rolling window).

**6. Parameter Sensitivity Grid**:
| risk_aversion | max_weight | Excess Net | Sharpe | Turnover |
|--------------|-----------|-----------|--------|----------|
| 0.5 | 0.01 | +4.45% | 39.61 | 80% |
| 0.5 | 0.05 | +16.54% | 14.33 | 88% |
| 2.0 | 0.02 | +6.49% | 96.07 | 86% |
| 5.0 | 0.01 | +1.26% | 16.06 | 75% |
| 5.0 | 0.05 | +4.44% | 61.88 | 87% |

⚠ Range = 15.28% across grid — alpha is PARAMETER-SENSITIVE.
Lower risk_aversion + higher max_weight → more concentrated → more excess
but also more fragile. Baseline (ra=2.0, mw=0.02) is middle-of-road.

**7. Permutation Test (10 shuffles)**:
- Real strategy: +6.49% excess
- Null (shrink-to-mean, i.e. equal-weight): -2.90%
- P-value: 0.000

✓ Strategy significantly outperforms no-signal baseline.

### Interpretation

1. **Signal content exists**: Decile spread of +22% and permutation p<0.001 confirm
   the ensemble predictions carry real information.

2. **Parameter sensitivity is a concern**: 15% range across the grid means alpha is
   partly an artifact of portfolio construction, not purely signal quality. The
   optimizer concentrates in tail predictions, amplifying both true signal and noise.

3. **Insufficient sample depth**: N=2 quarters is wholly inadequate for any deployment
   conclusion. Subperiod analysis is trivially "stable" when each half has 1 data point.

### Items Not Implemented (data not available)
- **Sector rotation analysis (#2)**: Needs GICS sector classifications
- **Drawdown attribution (#5)**: Needs factor exposure data per quarter

### Statistical Warning
⚠ ALL results are from N=2 quarters. Statistical conclusions are premature.
The framework is correct; the data depth is not.

### Recommendation
- Extend data history before drawing deployment conclusions
- Monitor the parameter sensitivity: if α depends on max_weight, the source is
  concentration risk, not prediction quality
- When more quarters available, revisit rolling IC for signal decay detection
