# Agent Instructions

ML-enhanced S&P 500 quarterly stock selection strategy. Ensemble of XGBoost, Ridge, and Random Forest with mean-variance optimization. Built on Python 3.14+ with `uv`.

## Quick Commands

```bash
uv sync                        # Install/sync dependencies
make all                       # Full pipeline: data → features → train
uv run python main.py --stage data      # Download data only
uv run python main.py --stage features  # Build features only
uv run python main.py --stage train     # Walk-forward backtest only
uv run pytest tests/ -v        # Run tests
make reports                   # Export report notebooks to HTML
uv run mkdocs serve            # Serve docs locally
```

## Architecture

All source code is in `src/stock_data/`. The codebase is **purely functional** — no classes in production code, all module-level functions, stateless.

### Data Flow

```
yfinance API → data/interim/*.parquet → data/processed/*.parquet → models/ → reports/figures/
```

### Module Responsibilities

| Module | Purpose |
|---|---|
| `config.py` | All constants, hyperparameters, and model params as module-level dicts |
| `dataset.py` | Data download (yfinance) and reshape into tidy `(symbol, date)` MultiIndex |
| `features.py` | ~180 features: profitability, size, balance sheet, cashflow, growth, momentum, macro, risk |
| `modeling/train.py` | Walk-forward backtest engine: expanding-window train → ensemble predict → MV optimize |
| `modeling/predict.py` | Pure-function helpers: winsorize, shrinkage, MV optimization (SLSQP), bootstrap CI |
| `evaluation.py` | Strategy evaluation, factor benchmarks, portfolio simulation |
| `plots.py` | Matplotlib diagnostic figures (returns `fig` objects) |
| `main.py` | Pipeline orchestration via `argparse --stage {data,features,train}` |

## Key Conventions

- **Index:** `(symbol, date)` MultiIndex is the canonical row index throughout. All joins rely on this.
- **Division safety:** Always `.replace(0, np.nan)` before division. Use `gcol(df, name, default)` for defensive column access (handles missing yfinance fields).
- **Serialization:** Parquet for DataFrames, `pickle` for lists/dicts (feature cols, weights, etc.).
- **Logging:** `print()` with formatted alignment for console reports. `logging` module is imported but rarely used.
- **No type hints** on most functions. Docstrings use brief Google-style.
- **Notebook naming:** `<step>.<version>-<initials>-<description>.ipynb` (Cookiecutter Data Science pattern). `exploratory/` for iteration, `reports/` for communication.

## Testing

Tests use **pytest with class-based grouping**. Only pure-computation functions are tested — no mocking of yfinance or network calls.

Pattern: helper factories (`_make_raw_dict`, `_basic_income_df`) build synthetic data, then assert properties (no inf, sum-to-one, bounds, index structure).

```bash
uv run pytest tests/ -v
```

## Documentation

Full docs built with MkDocs Material. See:
- [docs/strategy.md](docs/strategy.md) — Strategy specification (features, models, optimization)
- [docs/pipeline.md](docs/pipeline.md) — 8-stage pipeline with data flow diagram
- [docs/getting-started.md](docs/getting-started.md) — Setup and commands
- [docs/notebooks.md](docs/notebooks.md) — Notebook guide

API reference is auto-generated from docstrings via `mkdocstrings`.

## Pitfalls

- **yfinance rate limits:** `dataset.py` sleeps every 50 symbols. Full data download takes significant time.
- **Headless rendering:** `main.py` sets `matplotlib.use("Agg")` — do not remove or plots break in CI/headless environments.
- **Walk-forward leakage:** Training data must strictly precede test quarter. The 45-day `EARNINGS_LAG_DAYS` in `config.py` guards against lookahead bias.
- **Covariance fallback:** If Ledoit-Wolf fails (insufficient daily returns), the optimizer falls back to diagonal covariance using predicted volatilities.
- **RandomState vs default_rng:** `predict.py` uses `np.random.RandomState` for bootstrap reproducibility — do not migrate to `default_rng` without updating seeds.
