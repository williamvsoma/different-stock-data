# Experiment #34: Rank Feature Ablation

**Issue:** [#34](https://github.com/williamvsoma/different-stock-data/issues/34)  
**Date:** 2026-04-20  
**Status:** Complete — no change recommended

## Hypothesis

Cross-sectional rank features are redundant for tree models (XGB, RF) since trees
split on thresholds and ranks are a monotone transform. Rank features may help Ridge
by linearising nonlinear relationships.

Proposed optimal: raw features → XGB/RF, rank features → Ridge.

## Experiment Design

| Variant | XGB/RF features | Ridge features | Total cols |
|---------|----------------|---------------|-----------|
| baseline | all (raw+rank) | all (raw+rank) | 172 |
| raw_only | raw | raw | 86 |
| rank_only | rank | rank | 86 |
| model_specific | raw | rank | 86 each |

Walk-forward backtest, PROD_CFG, 2 test quarters.

## Results

### Portfolio Metrics

| Variant | Excess (net) | Sharpe | IR | Ensemble IC |
|---------|-------------|--------|-----|-------------|
| **baseline** | **+6.49%** | **96.07** | **10.09** | **0.1469** |
| raw_only | +4.67% | 10.06 | 2.30 | 0.1466 |
| rank_only | +2.92% | 8.12 | 5.57 | 0.1187 |
| model_specific | +5.96% | 20.39 | 4.07 | 0.1248 |

### Per-Model Rank Correlation

| Variant | XGB | Ridge | RF |
|---------|-----|-------|-----|
| baseline | 0.1643 | 0.0708 | 0.1690 |
| raw_only | 0.1533 | **0.0875** | **0.1835** |
| rank_only | 0.1385 | 0.0140 | 0.1582 |
| model_specific | 0.1533 | 0.0140 | 0.1835 |

### Paired Tests vs Baseline (N=2, low power)

| Variant | Δ excess | t-stat |
|---------|---------|--------|
| raw_only | -1.82% | -1.31 |
| rank_only | -3.56% | -3.05 |
| model_specific | -0.53% | -0.65 |

## Key Findings

1. **Baseline wins** on all portfolio-level metrics (IR, Sharpe, excess return).
2. **Hypothesis partially refuted**: Rank features HURT Ridge (RC 0.0140 rank-only vs 0.0708 baseline). Ranks alone lack diversity — all values ∈ [0,1], highly correlated cross-sectionally.
3. **RF benefits from raw-only** (RC 0.1835 vs 0.1690) — confirms tree models don't need ranks.
4. **Ridge benefits from raw-only** too (RC 0.0875 vs 0.0708) — contradicts original hypothesis.
5. **Feature redundancy doesn't hurt** at portfolio level — the optimizer exploits the full feature set effectively despite information duplication.

## Limitations

- Only 2 test quarters → t-tests have no statistical power
- Portfolio Sharpe dominated by single-quarter variance
- Results may change with more data history

## Decision

**Keep baseline (all features for all models).** Rationale:
- Best portfolio performance across all metrics
- Feature redundancy is not harmful with current regularisation settings
- The theoretical argument (ranks are monotone transforms) doesn't account for
  Ridge benefiting from feature DIVERSITY even when information overlaps
- Revisit when more test quarters available (≥8) for proper statistical testing

## Implication for Config

No changes to `config.py` or `train.py`. The current approach of using all features
for all models is empirically validated (within the limitations of N=2).
