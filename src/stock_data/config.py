"""Project-wide constants and configuration."""

# ── Data acquisition ───────────────────────────────────────────────────────────

EARNINGS_LAG_DAYS = 45  # conservative: most S&P 500 firms report within 40 days

MACRO_TICKERS = {
    "^VIX": "vix",
    "^TNX": "treasury_10y",
    "^IRX": "treasury_3m",
    "^GSPC": "sp500_level",
}

# ── Walk-forward production engine ─────────────────────────────────────────────

PROD_CFG = {
    "max_weight": 0.02,
    "shrinkage_alpha": 0.5,
    "winsor_pct": 0.05,
    "cost_bps": 20,
    "min_train_q": 3,
    "min_train_rows": 800,
    "min_test_stocks": 50,
    "risk_aversion": 2.0,
    "cov_lookback_days": 252,
    "max_train_q": 20,
    "feat_ratio_threshold": 0.3,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "random_state": 42,
}

RIDGE_PARAMS: dict[str, float] = {"alpha": 10.0}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": -1,
}

ENS_W = {"xgb": 0.5, "ridge": 0.25, "rf": 0.25}

# ── Simulation ─────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 1_000_000
COST_BPS = 20
WEIGHT_THRESHOLD = 0.001
N_BOOT = 10_000
