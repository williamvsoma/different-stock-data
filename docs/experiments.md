# Experiment Tracker

Track all experiments run against the strategy. Record hypothesis, method, result, and decision for each.

!!! warning "Pre-requisites"
    Experiments marked with ⛔ depend on fixing survivorship bias ([#17](https://github.com/williamvsoma/different-stock-data/issues/17)) and timing issues ([#18](https://github.com/williamvsoma/different-stock-data/issues/18), [#19](https://github.com/williamvsoma/different-stock-data/issues/19)) first. Results are invalid without clean data.

---

## EXP-01: Ensemble Weight Optimization

**Issue:** [#21](https://github.com/williamvsoma/different-stock-data/issues/21)
**Agent:** `senior-data-scientist`
**Status:** 🔲 Not started
**Depends on:** ⛔ #17, #18

| Detail | Value |
|---|---|
| **Hypothesis** | XGBoost 50% weight is arbitrary; Ridge may deserve more weight given noise level |
| **Method** | (a) Equal-weight 1/3 each (b) Inverse-variance from prior quarter IC (c) Stacking meta-learner (d) Solo model ablation |
| **Metric** | Ensemble rank correlation, net Sharpe, turnover, max drawdown |
| **Result** | — |
| **Decision** | — |

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
**Status:** 🔲 Not started

| Detail | Value |
|---|---|
| **Hypothesis** | Rank features are redundant for tree models; removing them halves dimensionality with no signal loss |
| **Method** | Walk-forward with (a) all features (b) raw only (c) ranks only. Per-model and ensemble comparison. |
| **Metric** | Per-model RC, ensemble RC, net Sharpe |
| **Result** | — |
| **Decision** | — |

---

## EXP-04: Volatility Model Benchmark

**Issue:** [#32](https://github.com/williamvsoma/different-stock-data/issues/32)
**Agent:** `senior-data-scientist`
**Status:** 🔲 Not started

| Detail | Value |
|---|---|
| **Hypothesis** | Historical vol may match or beat ML vol prediction; vol model doesn't need XGBoost |
| **Method** | (a) Separate vol model hyperparams (depth 5-7) (b) Naive `hist_vol_3m` as prediction (c) GARCH(1,1) baseline (d) Quality gate: fallback to historical if vol_rc < 0.2 |
| **Metric** | Vol rank correlation, portfolio Sharpe (since vol feeds into optimizer) |
| **Result** | — |
| **Decision** | — |

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

_None yet._
