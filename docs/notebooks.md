# Notebooks

The project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) convention:

- **Source code** (`src/stock_data/`) is for repetition — run `make all` to reproduce everything.
- **Notebooks** are for exploration and communication.

Naming convention: `<step>-<ghuser>-<description>.ipynb`

## Exploratory Notebooks

These notebooks inspect data and iterate on ideas. They are working documents, not polished outputs.

### 1.0 — Data Acquisition

**File**: `notebooks/exploratory/1.0-wvs-data-acquisition.ipynb`

Explores the raw data pipeline:

- Loading the S&P 500 universe
- Downloading quarterly financials via yfinance
- Inspecting data quality and coverage
- Reshaping statements into tidy format

### 2.0 — Feature Engineering

**File**: `notebooks/exploratory/2.0-wvs-feature-engineering.ipynb`

Develops and validates the feature set:

- Profitability, balance sheet, and cash flow ratios
- Momentum and macro features
- Cross-sectional ranking
- Feature correlation and distribution analysis

## Report Notebooks

Polished notebooks that present final results. Export to HTML with `make reports`.

### 3.0 — Backtest Results

**File**: `notebooks/reports/3.0-wvs-backtest-results.ipynb`

Presents the walk-forward backtest:

- Cumulative wealth curves vs benchmarks
- Per-quarter excess returns
- Model quality over time
- Feature importance stability

### 4.0 — Strategy Evaluation

**File**: `notebooks/reports/4.0-wvs-strategy-evaluation.ipynb`

Deep evaluation of strategy robustness:

- Factor benchmark comparison
- Bootstrap confidence intervals
- Regime dependence analysis
- Portfolio simulation with drawdowns
- External validity assessment
