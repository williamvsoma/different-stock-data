"""EXP-71: Co-optimize shrinkage_alpha, risk_aversion, max_weight.

Uses predict_all_quarters + optimize_from_predictions to sweep params
without retraining. Optimizes for net Sharpe ratio.
"""
import sys
from pathlib import Path
import pickle
import itertools
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_data.config import PROD_CFG
from stock_data.modeling.train import predict_all_quarters, optimize_from_predictions

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED = DATA_DIR / "data" / "processed"
INTERIM = DATA_DIR / "data" / "interim"


def main():
    risk_model_df = pd.read_parquet(PROCESSED / "risk_model_df.parquet")
    with open(PROCESSED / "feature_cols.pkl", "rb") as f:
        feature_cols_all = pickle.load(f)
    close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")

    print("Training models (one-time)...")
    predictions, _ = predict_all_quarters(
        risk_model_df, feature_cols_all, close_prices,
    )

    # Parameter grid
    grid = list(itertools.product(
        [0.25, 0.4, 0.5, 0.6, 0.75, 1.0],      # shrinkage_alpha
        [0.5, 1.0, 2.0, 3.0, 5.0],               # risk_aversion
        [0.01, 0.02, 0.03, 0.05],                 # max_weight
    ))

    results = []
    for sa, ra, mw in grid:
        cfg = {**PROD_CFG, "shrinkage_alpha": sa, "risk_aversion": ra, "max_weight": mw}
        df, _ = optimize_from_predictions(predictions, close_prices, cfg)
        ex = (df["net_ret"] - df["mkt_ret"]).mean()
        sr = df["net_ret"].mean() / df["net_ret"].std() if df["net_ret"].std() > 0 else 0
        results.append({
            "shrinkage_alpha": sa, "risk_aversion": ra, "max_weight": mw,
            "excess_net": ex, "sharpe": sr,
            "avg_turnover": df["turnover"].mean(),
            "n_held": df["n_held"].mean(),
        })

    df = pd.DataFrame(results)
    df.to_csv(Path(__file__).parent / "exp_71_results.csv", index=False)

    top5 = df.nlargest(5, "sharpe")
    print(f"\nTop 5 configurations by Sharpe ({len(grid)} total):")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
