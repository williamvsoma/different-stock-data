"""EXP-05 through EXP-10: Research experiments using modular walk-forward.

Leverages predict_all_quarters / optimize_from_predictions to run
optimizer-only sweeps without retraining.

Usage:
    uv run python experiments/exp_research.py --exp 05   # Subperiod stability
    uv run python experiments/exp_research.py --exp 06   # Decile monotonicity
    uv run python experiments/exp_research.py --exp 07   # Permutation test
    uv run python experiments/exp_research.py --exp 08   # Parameter sensitivity
    uv run python experiments/exp_research.py --exp 09   # Transaction cost sweep
    uv run python experiments/exp_research.py --exp all  # Run all
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_data.config import PROD_CFG
from stock_data.modeling.train import predict_all_quarters, optimize_from_predictions
from stock_data.modeling.predict import safe_spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED = DATA_DIR / "data" / "processed"
INTERIM = DATA_DIR / "data" / "interim"
OUT_DIR = Path(__file__).parent


def load_data():
    risk_model_df = pd.read_parquet(PROCESSED / "risk_model_df.parquet")
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols_all = json.load(f)
    close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")
    return risk_model_df, feature_cols_all, close_prices


def exp_05_subperiod(predictions, close_prices):
    """EXP-05: Subperiod stability — split into halves and report."""
    prod_df, _ = optimize_from_predictions(predictions, close_prices)
    n = len(prod_df)
    half = n // 2
    for label, sub in [("First half", prod_df.iloc[:half]),
                        ("Second half", prod_df.iloc[half:])]:
        ex = (sub["net_ret"] - sub["mkt_ret"]).mean()
        sr = sub["net_ret"].mean() / sub["net_ret"].std() if sub["net_ret"].std() > 0 else 0
        print(f"  {label}: N={len(sub)} | excess={ex:+.2%} | sharpe={sr:.2f}")


def exp_06_decile(predictions, close_prices):
    """EXP-06: Decile return monotonicity."""
    rows = []
    for pred in predictions:
        p_ens = pred["p_ens"]
        act_ret = pred["act_ret"]
        deciles = pd.qcut(p_ens, 10, labels=False, duplicates="drop")
        for d in sorted(set(deciles)):
            mask = deciles == d
            rows.append({"decile": d, "avg_ret": act_ret[mask].mean()})
    df = pd.DataFrame(rows).groupby("decile")["avg_ret"].mean()
    print("  Decile | Avg Realized Return")
    for d, r in df.items():
        print(f"  D{d:02d}    | {r:+.2%}")
    spread = df.iloc[-1] - df.iloc[0]
    print(f"\n  Long-short spread (D9-D0): {spread:+.2%}")
    df.to_csv(OUT_DIR / "exp_06_results.csv")


def exp_07_permutation(predictions, close_prices, n_perms=100):
    """EXP-07: Permutation test — compare real vs random signal."""
    real_df, _ = optimize_from_predictions(predictions, close_prices)
    real_sharpe = real_df["net_ret"].mean() / real_df["net_ret"].std() if real_df["net_ret"].std() > 0 else 0

    rng = np.random.default_rng(42)
    perm_sharpes = []
    for i in range(n_perms):
        perm_preds = []
        for pred in predictions:
            p = dict(pred)
            p["p_ens"] = rng.permutation(p["p_ens"])
            perm_preds.append(p)
        df, _ = optimize_from_predictions(perm_preds, close_prices)
        sr = df["net_ret"].mean() / df["net_ret"].std() if df["net_ret"].std() > 0 else 0
        perm_sharpes.append(sr)

    perm_sharpes = np.array(perm_sharpes)
    p_val = np.mean(perm_sharpes >= real_sharpe)
    print(f"  Real Sharpe: {real_sharpe:.2f}")
    print(f"  Permutation mean: {perm_sharpes.mean():.2f} ± {perm_sharpes.std():.2f}")
    print(f"  p-value: {p_val:.3f}")


def exp_08_param_grid(predictions, close_prices):
    """EXP-08: Parameter sensitivity grid."""
    results = []
    for ra in [0.5, 1.0, 2.0, 5.0]:
        for mw in [0.01, 0.02, 0.05, 0.10]:
            for sa in [0.25, 0.5, 0.75]:
                cfg = {**PROD_CFG, "risk_aversion": ra, "max_weight": mw,
                       "shrinkage_alpha": sa}
                df, _ = optimize_from_predictions(predictions, close_prices, cfg)
                ex = (df["net_ret"] - df["mkt_ret"]).mean()
                sr = df["net_ret"].mean() / df["net_ret"].std() if df["net_ret"].std() > 0 else 0
                results.append({"risk_aversion": ra, "max_weight": mw,
                                "shrinkage_alpha": sa, "excess_net": ex, "sharpe": sr})

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "exp_08_results.csv", index=False)
    best = df.loc[df["sharpe"].idxmax()]
    print(f"  Best: ra={best['risk_aversion']}, mw={best['max_weight']}, "
          f"sa={best['shrinkage_alpha']} → sharpe={best['sharpe']:.2f}")
    print(f"  Grid saved to experiments/exp_08_results.csv ({len(df)} combos)")


def exp_09_cost_sweep(predictions, close_prices):
    """EXP-09: Transaction cost sensitivity."""
    for bps in [10, 20, 30, 50, 100]:
        cfg = {**PROD_CFG, "cost_bps": bps}
        df, _ = optimize_from_predictions(predictions, close_prices, cfg)
        ex = (df["net_ret"] - df["mkt_ret"]).mean()
        sr = df["net_ret"].mean() / df["net_ret"].std() if df["net_ret"].std() > 0 else 0
        print(f"  {bps:3d} bps | excess={ex:+.2%} | sharpe={sr:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=["05", "06", "07", "08", "09", "all"])
    args = parser.parse_args()

    risk_model_df, feature_cols_all, close_prices = load_data()

    print("Training models (one-time)...")
    predictions, fi_list = predict_all_quarters(
        risk_model_df, feature_cols_all, close_prices,
    )

    experiments = {
        "05": ("EXP-05: Subperiod Stability", lambda: exp_05_subperiod(predictions, close_prices)),
        "06": ("EXP-06: Decile Monotonicity", lambda: exp_06_decile(predictions, close_prices)),
        "07": ("EXP-07: Permutation Test", lambda: exp_07_permutation(predictions, close_prices)),
        "08": ("EXP-08: Parameter Sensitivity", lambda: exp_08_param_grid(predictions, close_prices)),
        "09": ("EXP-09: Transaction Cost Sweep", lambda: exp_09_cost_sweep(predictions, close_prices)),
    }

    to_run = experiments.keys() if args.exp == "all" else [args.exp]
    for key in to_run:
        name, fn = experiments[key]
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        fn()


if __name__ == "__main__":
    main()
