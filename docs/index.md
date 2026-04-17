# S&P 500 Quarterly Stock Selection Strategy

ML-enhanced quarterly stock selection on S&P 500 constituents using an ensemble of XGBoost, Ridge, and Random Forest models with mean-variance optimization.

## Highlights

- **~180 features** spanning profitability, balance sheet, cash flow, momentum, macro, and risk
- **Ensemble modeling**: XGBoost (50%), Ridge (25%), Random Forest (25%)
- **Mean-variance optimization** with Ledoit-Wolf shrinkage covariance
- **Walk-forward validation** with factor benchmarks and bootstrap confidence intervals
- **Full reproducibility**: `make all` runs the entire pipeline from data download to report generation

## Quick Links

| Section | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation and first run |
| [Strategy Overview](strategy.md) | How the strategy works |
| [Pipeline](pipeline.md) | End-to-end pipeline stages |
| [API Reference](api/config.md) | Source code documentation |
| [Notebooks](notebooks.md) | Exploration and report notebooks |

## Project Layout

```
├── main.py                <- Full reproducible pipeline
├── data/
│   ├── raw/               <- Immutable source data (sp500_monthly.csv)
│   ├── interim/           <- Intermediate parquet files
│   └── processed/         <- Final feature matrix
├── models/                <- Trained model outputs & weight histories
├── notebooks/
│   ├── exploratory/       <- Data inspection & iteration
│   └── reports/           <- Polished results (exportable to HTML)
├── reports/figures/       <- Generated plots
└── src/stock_data/        <- Reusable source code
    ├── config.py          <- Constants & hyperparameters
    ├── dataset.py         <- Data download & reshaping
    ├── features.py        <- Feature engineering
    ├── evaluation.py      <- Strategy evaluation & simulation
    ├── plots.py           <- Visualization utilities
    └── modeling/
        ├── train.py       <- Walk-forward training engine
        └── predict.py     <- Optimization & prediction helpers
```
