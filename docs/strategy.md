# Strategy Overview

## Investment Thesis

Select S&P 500 stocks each quarter using ML-predicted returns and volatility, then allocate capital via mean-variance optimization. The strategy targets consistent risk-adjusted excess returns over an equal-weight market benchmark.

## Features (~180 total)

The feature set covers multiple dimensions of stock fundamentals, price behavior, and macro context:

| Category | Examples | Count |
|---|---|---|
| **Profitability** | Gross margin, operating margin, net margin, EBITDA margin, R&D intensity | ~10 |
| **Size** | Log revenue, log net income, diluted EPS | ~4 |
| **Balance Sheet** | ROE, ROA, ROIC, debt/equity, current ratio, asset turnover | ~16 |
| **Cash Flow** | OCF/revenue, FCF/revenue, cash conversion, capex/revenue | ~8 |
| **Growth (YoY)** | Annual revenue growth, net income growth, EBITDA growth | ~3 |
| **Growth (QoQ)** | Quarter-over-quarter revenue, margin, and EPS changes | ~8 |
| **Price Momentum** | 1m, 3m, 6m, 12m returns; mean reversion; relative strength | ~10+ |
| **Macro** | VIX, yield curve, S&P 500 momentum, volatility regime | ~10+ |
| **Risk** | Historical volatility, beta, idiosyncratic vol, downside deviation | ~10+ |
| **Cross-sectional Ranks** | Percentile ranks of key features within each quarter | ~100+ |

Features are engineered in [`stock_data.features.build_features()`](api/features.md).

## Models

### Return Prediction (Ensemble)

Three models predict next-quarter stock returns. Predictions are combined as a weighted ensemble:

| Model | Weight | Key Hyperparameters |
|---|---|---|
| **XGBoost** | 50% | 300 trees, depth 3, lr 0.05, subsample 0.8 |
| **Ridge** | 25% | α = 10.0 (standardized features) |
| **Random Forest** | 25% | 200 trees, depth 5, min leaf 20 |

Ensemble predictions are winsorized (5th–95th percentile) and shrunk toward the cross-sectional mean (α = 0.5) to reduce noise.

### Volatility Prediction

A separate XGBoost model (same hyperparameters) predicts next-quarter realized volatility. This feeds into the covariance estimate for portfolio optimization.

## Portfolio Construction

### Mean-Variance Optimization

The optimizer maximizes expected return minus a risk penalty:

$$\max_w \; w^\top \mu - \lambda \, w^\top \Sigma \, w$$

subject to $\sum w_i = 1$ and $0 \le w_i \le 2\%$.

- **Covariance**: Ledoit-Wolf shrinkage estimator from 252 days of daily returns (quarterly-scaled). Falls back to a diagonal covariance (from predicted volatilities) when insufficient price history exists.
- **Risk aversion**: λ = 2.0
- **Max position**: 2% per stock
- **Rebalance**: Quarterly, with a 45-day earnings lag

### Transaction Costs

One-way turnover is tracked each quarter. Transaction costs are modeled at **20 bps** round-trip and deducted from gross returns.

## Validation

### Walk-Forward Testing

The engine uses an expanding-window walk-forward protocol:

1. At each quarter $t$, train on all quarters before $t$
2. Predict returns and volatility for quarter $t$
3. Optimize portfolio weights and compute realized returns
4. Advance to quarter $t+1$

Minimum requirements: 3 training quarters, 400 training rows, 50 test stocks.

### Factor Benchmarks

Strategy performance is compared against four single-factor portfolios:

- **Low Volatility**: Sort on 6-month historical vol (ascending)
- **Momentum (3m)**: Sort on 3-month price momentum (descending)
- **Quality (ROE)**: Sort on return on equity (descending)
- **Value (FCF/Assets)**: Sort on free cash flow to assets (descending)

### Bootstrap Confidence Intervals

10,000 bootstrap resamples of quarterly excess returns produce a 95% confidence interval and the probability that true excess return is ≤ 0.
