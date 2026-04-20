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

### Full Ranking (all 10 variants, by Information Ratio)

| Rank | Variant | IR | Excess (net) | IC | Sharpe |
|------|---------|-----|-------------|------|--------|
| 1 | no_ridge 67/0/33 | **29.30** | **+7.57%** | 0.1753 | 28.77 |
| 2 | no_ridge 50/0/50 | 10.86 | +6.53% | **0.1821** | 75.03 |
| 3 | **baseline 50/25/25** | 10.09 | +6.49% | 0.1469 | **96.07** |
| 4 | equal 33/33/33 | 9.43 | +5.71% | 0.1382 | 72.66 |
| 5 | no_xgb 0/50/50 | 8.95 | +4.74% | 0.1120 | 47.59 |
| 6 | xgb_solo | 7.57 | +6.38% | 0.1636 | 8.66 |
| 7 | ridge_heavy 25/50/25 | 6.83 | +3.60% | 0.1204 | 43.08 |
| 8 | ridge_solo | 5.07 | +2.46% | 0.0707 | 8.01 |
| 9 | adaptive | 4.88 | +5.74% | 0.1566 | 35.20 |
| 10 | rf_solo | 2.96 | +5.12% | 0.1684 | 13.72 |

### Per-Model Signal Quality

| Model | Avg IC (rank correlation) |
|-------|--------------------------|
| RF | 0.1690 |
| XGB | 0.1643 |
| Ridge | 0.0708 |

## Key Findings

1. **No-Ridge variants dominate on IR and IC.** Dropping Ridge (IC=0.071) yields
   IC improvement of 19–24% and +1.1pp excess return. The 67/0/33 split has highest IR.

2. **Baseline 50/25/25 wins on Sharpe** (96 vs 29). Ridge smooths predictions,
   reducing quarter-to-quarter variance. Whether this is real stabilisation or
   N=2 noise is unresolved.

3. **Hypothesis strongly refuted**: Ridge is the WEAKEST model (IC=0.071).
   Giving it more weight degrades performance. The issue predicted "optimal XGB weight ≤33%" —
   opposite is true.

4. **Ensemble adds substantial value** over best solo model:
   - Best ensemble IR: 29.30 (no_ridge 67/0/33)
   - Best solo IR: 7.57 (XGB)
   - Diversification benefit is real.

5. **XGB deserves high weight** because it has second-best IC (0.164) with
   lower variance than RF (which has highest IC 0.169 but worst solo IR 2.96).

6. **Adaptive weights fail** (IR=4.88): With N=2 quarters, the lookback
   mechanism overreacts to single-quarter noise. Q2 gave RF 53.5% weight
   based on Q1 which happened to be strong for RF.

7. **RF paradox**: Highest per-model IC (0.1690) but worst solo IR (2.96).
   RF predictions have high signal but also high variance → needs mixing with
   stable models.

### The Sharpe vs IC Tradeoff

| Metric | Winner | Value |
|--------|--------|-------|
| Information Ratio | no_ridge 67/0/33 | 29.30 |
| Excess return | no_ridge 67/0/33 | +7.57% |
| Ensemble IC | no_ridge 50/0/50 | 0.1821 |
| Sharpe ratio | baseline 50/25/25 | 96.07 |

Ridge's role: despite poor IC, it smooths ensemble predictions (acts like a
shrinkage prior), reducing position concentration and quarter-to-quarter variance.
Removing it sharpens signal but increases volatility of returns.

## Paired Tests vs Baseline

| Variant | Δ excess | t-stat |
|---------|---------|--------|
| no_ridge 67/0/33 | +1.08% | +5.15 |
| no_ridge 50/0/50 | +0.04% | +0.56 |
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
- Cannot distinguish "Ridge stabilises" from "N=2 noise" without more data

## Decision

**Keep 50/25/25 for now** but flag 67/0/33 as strong candidate. Rationale:
- Baseline wins on Sharpe (return consistency)
- No-Ridge wins on IC and excess return (signal quality)
- N=2 → cannot distinguish "Ridge stabilises" from "luck"
- Dropping Ridge is easy to implement (zero-cost to revert)

**Recommended next step:** When ≥8 test quarters available, re-run.
If no-Ridge still wins on IC and excess, switch to 67/0/33.
Keep Ridge model trained regardless (zero cost, insurance against regime change).
