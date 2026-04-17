# Getting Started

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd different-stock-data

# Install dependencies
uv sync
```

## Running the Pipeline

The entire pipeline — data download, feature engineering, walk-forward backtest, and report generation — is a single command:

```bash
make all
# or equivalently:
uv run python main.py
```

This will:

1. Load the S&P 500 universe from `data/raw/sp500_monthly.csv`
2. Fetch quarterly financials, prices, and macro indicators via yfinance
3. Engineer ~180 features
4. Run a walk-forward backtest with an ensemble of ML models
5. Save artifacts to `data/interim/`, `data/processed/`, `models/`, and `reports/figures/`

## Exporting Reports

Convert report notebooks to standalone HTML files:

```bash
make reports
```

Output is written to `reports/`.

## Cleaning Up

```bash
# Remove Python cache files
make clean

# Remove ALL generated artifacts (data, models, reports)
make clean-all
```

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Ridge regression, Random Forest, Ledoit-Wolf covariance |
| xgboost | Gradient boosted trees |
| yfinance | Financial data download |
| scipy | Optimization and statistics |
| matplotlib | Plotting |
| pyarrow | Parquet file I/O |
| aiohttp | Async HTTP for yfinance |
