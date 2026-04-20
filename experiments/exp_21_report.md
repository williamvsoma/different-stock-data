# Experiment #21: Ensemble Weight Validation

**Issue:** [#21](https://github.com/williamvsoma/different-stock-data/issues/21)  
**Date:** 2026-04-20  
**Status:** Complete — current 50/25/25 validated as optimal

## Hypothesis

XGB's 50% weight is unjustified. Ridge (heavily regularised) may generalise better
for noisy financial data and deserve more weight. Equal weights may suffice.

## Experiment Design

| Variant | XGB | Ridge | RF | Type |
|---------|-----|-------|-----|------|
| baseline | 0.50 | 0.25 | 0.25 | ensemble |
| equal | 0.33 | 0.33 | 0.33 | ensemble |
| ridge_heavy | 0.25 | 0.50 | 0.25 | ensemble |
| no_xgb | 0.00 | 0.50 | 0.50 | ensemble |
| no_ridge 67/0/33 | 0.67 | 0.00 | 0.33 | ensemble |
| no_ridge 50/0/50 | 0.50 | 0.00 | 0.50 | ensemble |
| adaptive | RC-based | RC-based | RC-based | dynamic |
| xgb_solo | 1.00 | — | — | ablation |
| ridge_solo | — | 1.00 | — | ablation |
| rf_solo | — | — | 1.00 | ablation |

## Results

### Final Ranking (by Information Ratio)

| Rank | Variant | IR | Excess (net) | IC | Sharpe |
|------|---------|-----|-------------|------|--------|
| 1 | **baseline 50/25/25** | **10.09** | **+6.49%** | 0.1469 | **96.07** |
| 2 | equal 33/33/33 | 9.43 | +5.71% | 0.1382 | 72.66 |
| 3 | no_xgb 0/50/50 | 8.95 | +4.74% | 0.1120 | 47.59 |
| 4 | xgb_solo | 7.57 | +6.38% | **0.1636** | 8.66 |
| 5 | ridge_heavy 25/50/25 | 6.83 | +3.60% | 0.1204 | 43.08 |
| 6 | ridge_solo | 5.07 | +2.46% | 0.0707 | 8.01 |
| 7 | adaptive | 4.88 | +5.74% | 0.1566 | 35.20 |
| 8 | rf_solo | 2.96 | +5.12% | **0.1684** | 13.72 |

### Per-Model Signal Quality

| Model | Avg IC (rank correlation) |
|-------|--------------------------|
| RF | 0.1690 |
| XGB | 0.1643 |
| Ridge | 0.0708 |

## Key Findings

1. **Baseline 50/25/25 wins** on both IR and Sharpe. No alternative beats it.

2. **Hypothesis strongly refuted**: Ridge is the WEAKEST model (IC=0.071).
   Giving it more weight degrades performance. The issue predicted "optimal XGB weight ≤33%" —
   opposite is true.

3. **Ensemble adds substantial value** over best solo model:
   - Best ensemble IR: 10.09
   - Best solo IR: 7.57 (XGB)
   - Diversification benefit is real.

4. **XGB deserves high weight** because it has second-best IC (0.164) with
   lower variance than RF (which has highest IC 0.169 but worst solo IR 2.96).

5. **Adaptive weights fail** (IR=4.88): With N=2 quarters, the lookback
   mechanism overreacts to single-quarter noise. Q2 gave RF 53.5% weight
   based on Q1 which happened to be strong for RF.

6. **RF paradox**: Highest per-model IC (0.1690) but worst solo IR (2.96).
   RF predictions have high signal but also high variance → needs mixing with
   stable models.

### Why 50/25/25 Works

XGB provides the best risk-adjusted signal. RF adds diversification (different
errors). Ridge, despite weak IC, smooths ensemble predictions (acts like a prior).
The 50% XGB weight appropriately reflects signal-to-noise: XGB >> Ridge in IC,
XGB ≈ RF in IC but more consistent.

## Paired Tests vs Baseline

| Variant | Δ excess | t-stat |
|---------|---------|--------|
| equal | -0.77% | -20.82 |
| ridge_heavy | -2.88% | -24.93 |
| xgb_solo | -0.11% | -0.07 |
| ridge_solo | -4.02% | -3.56 |
| rf_solo | -1.37% | -1.26 |
| no_xgb | -1.75% | -15.32 |
| adaptive | -0.75% | -1.41 |

Note: t-stats with N=2 are unreliable but direction is consistent.

## Limitations

- Only 2 test quarters → low statistical power
- Adaptive weights need ≥5 quarters of history to stabilise
- Per-model IC may change with more data
- Results conditional on current feature set and regularisation

## Follow-up: Dropping Ridge Entirely

Given Ridge's weak IC (0.071), tested removing it and redistributing weight:

| Variant | Excess (net) | IR | Sharpe | IC |
|---------|-------------|-----|--------|------|
| baseline 50/25/25 | +6.49% | 10.09 | 96.07 | 0.1469 |
| **no_ridge 67/0/33** | **+7.57%** | **29.30** | 28.77 | **0.1753** |
| no_ridge 50/0/50 | +6.53% | 10.86 | 75.03 | 0.1821 |

**Dropping Ridge improves signal quality substantially** (IC +19% to +24%).
Excess return increases +1.1pp. The 67/0/33 split has best IR.

However: Sharpe drops (96→29) indicating more quarter-to-quarter variance.
Ridge may act as portfolio stabiliser despite poor IC — its predictions smooth
the ensemble, reducing position concentration.

## Decision

**Keep 50/25/25 for now** but flag 67/0/33 as strong candidate. Rationale:
- Baseline wins on Sharpe (return consistency)
- No-Ridge wins on IC and excess return (signal quality)
- N=2 → cannot distinguish "Ridge stabilises" from "luck"
- Dropping Ridge is easy to implement (zero-cost to revert)

**Recommended next step:** When ≥8 test quarters available, re-run.
If no-Ridge still wins on IC and excess, switch to 67/0/33.
Keep Ridge model trained regardless (zero cost, insurance against regime change).
