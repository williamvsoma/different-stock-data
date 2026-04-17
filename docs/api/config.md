# Configuration

Project-wide constants and hyperparameters.

::: stock_data.config
    options:
      show_source: true
      members_order: source

## Key Parameters

### Data Acquisition

| Parameter | Value | Description |
|---|---|---|
| `EARNINGS_LAG_DAYS` | 45 | Days after quarter-end before trading (conservative reporting window) |
| `MACRO_TICKERS` | VIX, TNX, IRX, GSPC | Macro indicators downloaded from yfinance |

### Walk-Forward Engine

| Parameter | Value | Description |
|---|---|---|
| `max_weight` | 2% | Maximum portfolio weight per stock |
| `shrinkage_alpha` | 0.5 | Prediction shrinkage toward cross-sectional mean |
| `winsor_pct` | 5% | Winsorization percentile for predictions |
| `cost_bps` | 20 | Transaction cost per rebalance (basis points) |
| `min_train_q` | 3 | Minimum training quarters required |
| `min_train_rows` | 400 | Minimum training samples required |
| `min_test_stocks` | 50 | Minimum stocks in test set |
| `risk_aversion` | 2.0 | Mean-variance risk aversion parameter (λ) |
| `cov_lookback_days` | 252 | Days of price history for covariance estimation |

### Model Hyperparameters

**XGBoost** (`XGB_PARAMS`): 300 estimators, max depth 3, learning rate 0.05, subsample 0.8, colsample 0.7.

**Ridge** (`RIDGE_PARAMS`): α = 10.0.

**Random Forest** (`RF_PARAMS`): 200 estimators, max depth 5, min leaf 20, max features 0.5.

**Ensemble weights** (`ENS_W`): XGBoost 50%, Ridge 25%, Random Forest 25%.

### Simulation

| Parameter | Value | Description |
|---|---|---|
| `INITIAL_CAPITAL` | $1,000,000 | Starting portfolio value |
| `COST_BPS` | 20 | Transaction costs |
| `WEIGHT_THRESHOLD` | 0.001 | Minimum weight to count as a holding |
| `N_BOOT` | 10,000 | Bootstrap resamples for confidence intervals |
