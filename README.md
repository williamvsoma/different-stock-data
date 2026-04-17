# S&P 500 Quarterly Stock Selection Strategy

ML-enhanced quarterly stock selection on S&P 500 constituents using an ensemble of XGBoost, Ridge, and Random Forest models with mean-variance optimization.

## Project Organization

```
├── Makefile               <- Makefile with convenience commands
├── README.md
├── main.py                <- Full reproducible pipeline (data → model → report)
├── data
│   ├── external           <- Data from third party sources
│   ├── interim            <- Intermediate transformed data (parquet)
│   ├── processed          <- Final canonical datasets for modeling
│   └── raw                <- Original immutable data (e.g. sp500_monthly.csv)
├── docs                   <- Documentation
├── models                 <- Trained model outputs and weight histories
├── notebooks
│   ├── exploratory        <- Initial explorations and data inspection
│   │   ├── 1.0-wvs-data-acquisition.ipynb
│   │   └── 2.0-wvs-feature-engineering.ipynb
│   └── reports            <- Polished analysis, exportable to reports/
│       ├── 3.0-wvs-backtest-results.ipynb
│       └── 4.0-wvs-strategy-evaluation.ipynb
├── pyproject.toml         <- Project configuration & dependencies
├── references             <- Data dictionaries, manuals, etc.
├── reports
│   ├── figures            <- Generated graphics and figures
│   └── *.html             <- Exported report notebooks
└── src/stock_data         <- Source code for repetition & testing
    ├── __init__.py
    ├── config.py           <- Constants & hyperparameters
    ├── dataset.py          <- Data download & reshaping (yfinance)
    ├── evaluation.py       <- Strategy evaluation & simulation
    ├── features.py         <- Feature engineering
    ├── plots.py            <- Visualization utilities
    └── modeling
        ├── __init__.py
        ├── predict.py      <- Optimization & prediction helpers
        └── train.py        <- Walk-forward training engine
```

Notebook naming convention: `<step>-<ghuser>-<description>.ipynb`

## Quickstart

```bash
# Install dependencies (using uv)
uv sync

# Run the full reproducible pipeline
make all          # or: uv run python main.py

# Export report notebooks to HTML
make reports
```

## Workflow

The project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) philosophy:

- **Source code is for repetition.** All pipeline logic lives in `src/stock_data/`.
  Run `make all` to reproduce the entire analysis from scratch.
- **Notebooks are for exploration and communication.** Exploratory notebooks
  (in `notebooks/exploratory/`) inspect and iterate on data. Report notebooks
  (in `notebooks/reports/`) present polished results and can be exported as
  HTML to the `reports/` directory with `make reports`.
- **Data flows through stages.** Raw data in `data/raw/` is immutable. Interim
  artifacts go to `data/interim/`. The final feature matrix lives in
  `data/processed/`. Model outputs are saved to `models/`.

## Data

The strategy uses:
- **S&P 500 membership** from `data/raw/sp500_monthly.csv`
- **Quarterly financials** (income, balance sheet, cash flow) via yfinance
- **Daily prices** and **macro indicators** (VIX, Treasury yields) via yfinance

> **⚠️ Data Quality Notice:** yfinance provides *restated* financial statements
> without point-in-time tracking. This means backtest results use data that may
> differ from what was available at the time of each historical decision. The
> 45-day `EARNINGS_LAG_DAYS` guard mitigates timing bias but cannot address
> restatement bias. See [docs/strategy.md](docs/strategy.md) for details and
> mitigation plans.

## Strategy

1. **Features**: ~180 features including profitability ratios, balance sheet metrics, cash flow ratios, YoY/QoQ growth, price momentum, macro indicators, risk features, and cross-sectional ranks.
2. **Models**: Ensemble of XGBoost (50%), Ridge (25%), and Random Forest (25%) for return prediction; XGBoost for volatility prediction.
3. **Portfolio**: Mean-variance optimization with Ledoit-Wolf shrinkage covariance, 2% max position, quarterly rebalance with 45-day earnings lag, 20bps transaction costs.
4. **Validation**: Walk-forward out-of-sample evaluation, factor benchmarks, bootstrap confidence intervals.
