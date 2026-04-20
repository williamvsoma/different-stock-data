"""Data acquisition: download financials, prices, and macro data from yfinance."""

import logging
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from stock_data.config import EARNINGS_LAG_DAYS, MACRO_TICKERS

logger = logging.getLogger(__name__)


# ── S&P 500 universe ──────────────────────────────────────────────────────────


def load_sp500_universe(path: str) -> pd.DataFrame:
    """Load the full S&P 500 membership table (current and historical members)."""
    sp = pd.read_csv(path, parse_dates=["date_added", "date_removed"])
    return sp


def get_active_symbols(sp500_df: pd.DataFrame) -> list[str]:
    return sp500_df["symbol"].unique().tolist()


def get_universe_at_date(sp500_df: pd.DataFrame, as_of_date) -> list[str]:
    """Return symbols that were in the S&P 500 as of *as_of_date*."""
    as_of_date = pd.Timestamp(as_of_date)
    mask = (
        (sp500_df["date_added"] <= as_of_date)
        & (sp500_df["date_removed"].isna() | (sp500_df["date_removed"] > as_of_date))
    )
    return sp500_df.loc[mask, "symbol"].unique().tolist()


def filter_by_membership(df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (symbol, date) rows where the symbol was in the S&P 500 at that date.

    *df* must have a ``(symbol, date)`` MultiIndex.
    Vectorized: builds membership sets per unique date, then uses set lookups.
    """
    if len(df) == 0:
        return df
    idx = df.index.to_frame(index=False)
    unique_dates = idx["date"].unique()
    membership = {d: set(get_universe_at_date(sp500_df, d)) for d in unique_dates}
    mask = np.array([sym in membership[dt] for sym, dt in zip(idx["symbol"], idx["date"])])
    return df.loc[mask]


# ── Financial statements ───────────────────────────────────────────────────────


def _fetch_statements(symbols, accessor, label, sleep_every=50, max_retries=2):
    """Generic fetcher for quarterly/annual yfinance statement attributes."""
    data = {}
    failed = []
    t0 = time.time()
    for i, symbol in enumerate(symbols):
        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                stmt = getattr(ticker, accessor)
                if stmt is not None and not stmt.empty:
                    data[symbol] = stmt
                break
            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                if attempt == max_retries:
                    failed.append((symbol, str(e)))
                    logger.warning("Failed to fetch %s for %s after %d attempts: %s",
                                   label, symbol, max_retries, e)
                else:
                    time.sleep(2 ** attempt)
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


def download_prices(symbols, start, end) -> tuple[pd.DataFrame, list[str]]:
    """Download daily close prices and return (tidy DataFrame, failed_symbols).

    Raises RuntimeError if >10% of symbols fail to download.
    """
    print(f"  Downloading daily prices for {len(symbols)} symbols "
          f"({start.date()} to {end.date()})...")
    # Include ^GSPC for cap-weighted benchmark
    dl_symbols = list(symbols) + ["^GSPC"]
    prices = yf.download(dl_symbols, start=start, end=end,
                         interval="1d", group_by="ticker", threads=True)
    frames = []
    successful = set()
    for sym in dl_symbols:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                s = prices[(sym, "Close")].dropna()
            else:
                s = prices["Close"].dropna()
            if len(s) > 0:
                frames.append(
                    pd.DataFrame({"date": s.index, "symbol": sym, "close": s.values})
                )
                successful.add(sym)
        except (KeyError, TypeError):
            pass
    failed = [sym for sym in dl_symbols if sym not in successful]
    close = pd.concat(frames, ignore_index=True)
    close["date"] = pd.to_datetime(close["date"])
    n_requested = len(symbols)  # exclude ^GSPC from coverage calc
    n_failed_stocks = len([s for s in failed if s != "^GSPC"])
    coverage_pct = (n_requested - n_failed_stocks) / n_requested * 100
    print(f"  close_prices: {len(close):,} rows, "
          f"{close['symbol'].nunique()} symbols "
          f"(coverage: {coverage_pct:.1f}%, {n_failed_stocks} failed)")
    if n_failed_stocks > 0:
        print(f"  Failed symbols: {failed[:20]}{'...' if len(failed) > 20 else ''}")
    if n_failed_stocks > n_requested * 0.10:
        raise RuntimeError(
            f">{10}% of symbols failed price download: "
            f"{n_failed_stocks}/{n_requested} ({100-coverage_pct:.1f}%)"
        )
    return close, failed


def build_returns_panel(close_prices):
    """Pre-compute a (date × symbol) daily returns panel.

    Returns a dict with keys:
    - 'close_panel': DataFrame (date × symbol) of close prices
    - 'returns_panel': DataFrame (date × symbol) of daily returns
    - 'grouped': dict {symbol: sorted Series of close prices indexed by date}

    This avoids redundant groupby/pivot in momentum_features and risk_features.
    """
    close_panel = close_prices.pivot_table(
        index="date", columns="symbol", values="close",
    )
    close_panel = close_panel.sort_index()
    returns_panel = close_panel.pct_change()

    # Pre-grouped for per-symbol lookups (still needed by some code paths)
    grouped = {
        sym: close_panel[sym].dropna()
        for sym in close_panel.columns
    }

    return {
        "close_panel": close_panel,
        "returns_panel": returns_panel,
        "grouped": grouped,
    }


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
    """Compute next-quarter returns with EARNINGS_LAG_DAYS delay.

    Vectorized: uses merge_asof for buy/sell price lookups instead of
    per-symbol per-quarter Python loops.
    """
    # Build (symbol, q_date) frame from MultiIndex
    pairs = sym_date_pairs.to_frame(index=False)
    pairs["buy_target"] = pairs["date"] + pd.Timedelta(days=EARNINGS_LAG_DAYS)
    pairs["sell_target"] = pairs["buy_target"] + pd.DateOffset(months=3)

    cp = close_prices.sort_values(["symbol", "date"])

    # Prepare sorted copies for buy and sell lookups
    # merge_asof requires both sides sorted by the merge key
    cp_buy = cp.rename(columns={"date": "buy_date", "close": "buy_price"}).sort_values("buy_date")
    cp_sell = cp.rename(columns={"date": "sell_date", "close": "sell_price"}).sort_values("sell_date")

    # merge_asof for buy prices: first trade on or after buy_target (within 10 days)
    buy = pd.merge_asof(
        pairs.sort_values("buy_target"),
        cp_buy,
        left_on="buy_target", right_on="buy_date",
        by="symbol", direction="forward", tolerance=pd.Timedelta(days=10),
    )

    # merge_asof for sell prices: first trade on or after sell_target (within 10 days)
    sell = pd.merge_asof(
        pairs.sort_values("sell_target"),
        cp_sell,
        left_on="sell_target", right_on="sell_date",
        by="symbol", direction="forward", tolerance=pd.Timedelta(days=10),
    )

    # Join buy and sell on (symbol, date)
    merged = buy.merge(
        sell[["symbol", "date", "sell_date", "sell_price"]],
        on=["symbol", "date"], how="inner",
    )
    valid = merged["buy_price"].notna() & merged["sell_price"].notna()
    merged = merged[valid].copy()
    merged["next_q_return"] = merged["sell_price"] / merged["buy_price"] - 1
    return merged[["symbol", "date", "buy_date", "sell_date", "next_q_return"]]


# ── Realized volatility ───────────────────────────────────────────────────────


def compute_realized_vol(returns_df, close_prices) -> pd.DataFrame:
    """Compute realized forward volatility for each (symbol, quarter).

    Vectorized: groups daily prices by symbol once, then computes
    statistics for each holding period via vectorized slice operations.
    """
    grouped = {
        sym: grp.sort_values("date").set_index("date")["close"]
        for sym, grp in close_prices.groupby("symbol")
    }
    records = []
    for _, row in returns_df.iterrows():
        sym, buy_dt, sell_dt = row["symbol"], row["buy_date"], row["sell_date"]
        series = grouped.get(sym)
        if series is None:
            continue
        daily = series.loc[buy_dt:sell_dt]
        if len(daily) < 10:
            continue
        dr = daily.pct_change().dropna()
        neg = dr[dr < 0]
        cum = daily.values
        records.append({
            "symbol": sym,
            "date": row["date"],
            "realized_vol": dr.std() * np.sqrt(252),
            "realized_downside_vol": neg.std() * np.sqrt(252) if len(neg) > 0 else np.nan,
            "realized_max_dd": (cum / np.maximum.accumulate(cum) - 1).min(),
            "realized_var5": dr.quantile(0.05),
            "realized_skew": dr.skew(),
            "realized_kurtosis": dr.kurtosis(),
        })
    return pd.DataFrame(records)


# ── Data quality checks ───────────────────────────────────────────────────────


def validate_data_quality(wide_df, expected_min_quarters=4):
    """Flag symbols with fewer quarters than expected.

    Returns a DataFrame with per-symbol quarter count and a warning flag.
    Prints a summary of flagged symbols.
    """
    counts = wide_df.groupby(level="symbol").size().rename("n_quarters")
    quality = counts.to_frame()
    quality["flagged"] = quality["n_quarters"] < expected_min_quarters
    n_flagged = quality["flagged"].sum()
    n_total = len(quality)
    print(f"  Data quality: {n_total} symbols, "
          f"{n_flagged} with <{expected_min_quarters} quarters")
    if n_flagged > 0:
        flagged_syms = quality[quality["flagged"]].index.tolist()
        print(f"    Flagged: {flagged_syms[:20]}"
              + (f" ... (+{n_flagged - 20} more)" if n_flagged > 20 else ""))
    return quality
