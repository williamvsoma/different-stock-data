"""Data acquisition: download financials, prices, and macro data from yfinance."""

import time

import numpy as np
import pandas as pd
import yfinance as yf

from stock_data.config import EARNINGS_LAG_DAYS, MACRO_TICKERS


# ── S&P 500 universe ──────────────────────────────────────────────────────────


def load_sp500_universe(path: str) -> pd.DataFrame:
    """Load the S&P 500 membership table and return currently active symbols."""
    sp = pd.read_csv(path, parse_dates=["date_added", "date_removed"])
    current = sp[sp["date_removed"].isna()]
    return current


def get_active_symbols(sp500_df: pd.DataFrame) -> list[str]:
    return sp500_df["symbol"].unique().tolist()


# ── Financial statements ───────────────────────────────────────────────────────


def _fetch_statements(symbols, accessor, label, sleep_every=50):
    """Generic fetcher for quarterly/annual yfinance statement attributes."""
    data = {}
    failed = []
    t0 = time.time()
    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            stmt = getattr(ticker, accessor)
            if stmt is not None and not stmt.empty:
                data[symbol] = stmt
        except Exception as e:
            failed.append((symbol, str(e)))
        if (i + 1) % sleep_every == 0:
            print(f"  {label}: {i + 1}/{len(symbols)}...")
            time.sleep(1)
    elapsed = time.time() - t0
    print(f"  {label}: done in {elapsed:.1f}s — {len(data)} symbols "
          f"({len(failed)} failed)")
    return data, failed


def fetch_quarterly_income(symbols):
    return _fetch_statements(symbols, "quarterly_income_stmt",
                             "Quarterly income")


def fetch_quarterly_balance_sheets(symbols):
    return _fetch_statements(symbols, "quarterly_balance_sheet",
                             "Quarterly balance sheet")


def fetch_quarterly_cashflows(symbols):
    return _fetch_statements(symbols, "quarterly_cashflow",
                             "Quarterly cash flow")


def fetch_annual_income(symbols):
    return _fetch_statements(symbols, "income_stmt", "Annual income")


# ── Reshape helpers ────────────────────────────────────────────────────────────


def reshape_statements(raw_dict: dict) -> pd.DataFrame:
    """Stack a {symbol: wide_df} dict into a tidy (symbol, date, item) table."""
    combined = (
        pd.concat(
            {s: df for s, df in raw_dict.items()},
            names=["symbol", "item"],
            sort=True,
        )
        .stack()
        .rename_axis(["symbol", "item", "date"])
        .rename("value")
        .reset_index()
        .set_index(["symbol", "date", "item"])
        .sort_index()
    )
    return combined


def drop_sparse_pairs(combined: pd.DataFrame, threshold: float = 0.5):
    """Drop (symbol, date) groups where >threshold fraction of values are NaN."""
    nan_frac = combined.groupby(level=["symbol", "date"])["value"].apply(
        lambda s: s.isna().mean()
    )
    pairs_to_drop = nan_frac[nan_frac > threshold].index
    before = len(combined)
    combined = combined.drop(index=pairs_to_drop)
    print(f"  Dropped {len(pairs_to_drop)} >50%-NaN (symbol,date) pairs "
          f"({before:,} → {len(combined):,} rows)")
    return combined


def pivot_statements(combined: pd.DataFrame) -> pd.DataFrame:
    """Pivot a stacked statement table into wide (symbol, date) × items."""
    wide = combined.reset_index().pivot_table(
        index=["symbol", "date"], columns="item", values="value"
    )
    wide.columns.name = None
    return wide


def reshape_annual_income(annual_dict: dict) -> pd.DataFrame:
    """Reshape annual income statements into (symbol, date) × items."""
    frames = []
    for sym, df in annual_dict.items():
        for col_name in df.columns:
            row = df[col_name].to_frame().T
            row.index = [col_name]
            row["symbol"] = sym
            row["date"] = col_name
            frames.append(row)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True).set_index(["symbol", "date"]).sort_index()
    return raw.select_dtypes(include=[np.number])


# ── Price data ─────────────────────────────────────────────────────────────────


def download_prices(symbols, start, end) -> pd.DataFrame:
    """Download daily close prices and return a tidy DataFrame."""
    print(f"  Downloading daily prices for {len(symbols)} symbols "
          f"({start.date()} to {end.date()})...")
    prices = yf.download(symbols, start=start, end=end,
                         interval="1d", group_by="ticker", threads=True)
    frames = []
    for sym in symbols:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                s = prices[(sym, "Close")].dropna()
            else:
                s = prices["Close"].dropna()
            if len(s) > 0:
                frames.append(
                    pd.DataFrame({"date": s.index, "symbol": sym, "close": s.values})
                )
        except (KeyError, TypeError):
            pass
    close = pd.concat(frames, ignore_index=True)
    close["date"] = pd.to_datetime(close["date"])
    print(f"  close_prices: {len(close):,} rows, "
          f"{close['symbol'].nunique()} symbols")
    return close


# ── Macro data ─────────────────────────────────────────────────────────────────


def download_macro(start, end) -> pd.DataFrame:
    """Download macro indicators and derive features."""
    raw = yf.download(
        list(MACRO_TICKERS.keys()), start=start, end=end,
        interval="1d", threads=True, progress=False,
    )
    frames = []
    for ticker_sym, col_name in MACRO_TICKERS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw[("Close", ticker_sym)].dropna()
            else:
                s = raw["Close"].dropna()
            if len(s) > 0:
                frames.append(pd.DataFrame({col_name: s.values}, index=s.index))
        except (KeyError, TypeError):
            print(f"    Warning: {ticker_sym} ({col_name}) not available")

    macro = pd.concat(frames, axis=1).sort_index()

    # Derived features
    macro["yield_curve"] = macro.get("treasury_10y", 0) - macro.get("treasury_3m", 0)
    if "vix" in macro.columns:
        macro["vix_20d_ma"] = macro["vix"].rolling(20).mean()
        macro["vix_regime"] = (macro["vix"] > macro["vix_20d_ma"]).astype(float)
        macro["vix_change_20d"] = macro["vix"].pct_change(20)
    if "sp500_level" in macro.columns:
        macro["sp500_ret_60d"] = macro["sp500_level"].pct_change(60)
        macro["sp500_ret_20d"] = macro["sp500_level"].pct_change(20)
        macro["sp500_vol_20d"] = (
            macro["sp500_level"].pct_change().rolling(20).std() * np.sqrt(252)
        )
    macro = macro.drop(columns=["sp500_level"], errors="ignore")
    return macro


# ── Forward returns ────────────────────────────────────────────────────────────


def compute_forward_returns(close_prices, sym_date_pairs) -> pd.DataFrame:
    """Compute next-quarter returns with EARNINGS_LAG_DAYS delay."""
    records = []
    for symbol in close_prices["symbol"].unique():
        grp = close_prices[close_prices["symbol"] == symbol].set_index("date").sort_index()
        if grp.empty:
            continue
        stock_dates = (
            sym_date_pairs[sym_date_pairs.get_level_values("symbol") == symbol]
            .get_level_values("date")
        )
        for q_date in stock_dates:
            buy_date = q_date + pd.Timedelta(days=EARNINGS_LAG_DAYS)
            sell_date = buy_date + pd.DateOffset(months=3)
            buy_w = grp.loc[
                buy_date - pd.Timedelta(days=5):buy_date + pd.Timedelta(days=5), "close"
            ]
            sell_w = grp.loc[
                sell_date - pd.Timedelta(days=5):sell_date + pd.Timedelta(days=5), "close"
            ]
            if len(buy_w) > 0 and len(sell_w) > 0:
                ret = sell_w.iloc[0] / buy_w.iloc[0] - 1
                records.append({
                    "symbol": symbol,
                    "date": q_date,
                    "buy_date": buy_w.index[0],
                    "sell_date": sell_w.index[0],
                    "next_q_return": ret,
                })
    return pd.DataFrame(records)


# ── Realized volatility ───────────────────────────────────────────────────────


def compute_realized_vol(returns_df, close_prices) -> pd.DataFrame:
    """Compute realized forward volatility for each (symbol, quarter)."""
    records = []
    for _, row in returns_df.iterrows():
        sym, buy_dt, sell_dt = row["symbol"], row["buy_date"], row["sell_date"]
        daily = close_prices[
            (close_prices["symbol"] == sym)
            & (close_prices["date"] >= buy_dt)
            & (close_prices["date"] <= sell_dt)
        ].sort_values("date")
        if len(daily) < 10:
            continue
        dr = daily["close"].pct_change().dropna()
        records.append({
            "symbol": sym,
            "date": row["date"],
            "realized_vol": dr.std() * np.sqrt(252),
            "realized_downside_vol": dr[dr < 0].std() * np.sqrt(252),
            "realized_max_dd": (daily["close"] / daily["close"].cummax() - 1).min(),
            "realized_var5": dr.quantile(0.05),
            "realized_skew": dr.skew(),
            "realized_kurtosis": dr.kurtosis(),
        })
    return pd.DataFrame(records)
