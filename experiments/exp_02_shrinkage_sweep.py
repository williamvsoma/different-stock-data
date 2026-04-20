"""EXP-02: Shrinkage Alpha Sweep.

Tests α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} by re-running optimization only
(no retraining) using predict_all_quarters / optimize_from_predictions.
"""
import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_data.config import PROD_CFG
from stock_data.modeling.train import predict_all_quarters, optimize_from_predictions

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED = DATA_DIR / "data" / "processed"
INTERIM = DATA_DIR / "data" / "interim"


def main():
    risk_model_df = pd.read_parquet(PROCESSED / "risk_model_df.parquet")
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols_all = json.load(f)
    close_prices = pd.read_parquet(INTERIM / "close_prices.parquet")

    # Train once
    print("Training models (one-time)...")
    predictions, fi_list = predict_all_quarters(
        risk_model_df, feature_cols_all, close_prices,
    )

    # Sweep shrinkage_alpha
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    for alpha in alphas:
        cfg = {**PROD_CFG, "shrinkage_alpha": alpha}
        prod_df, _ = optimize_from_predictions(predictions, close_prices, cfg)
        ex_net = (prod_df["net_ret"] - prod_df["mkt_ret"]).mean()
        sharpe = prod_df["net_ret"].mean() / prod_df["net_ret"].std() if prod_df["net_ret"].std() > 0 else 0
        results.append({
            "shrinkage_alpha": alpha,
            "avg_excess_net": ex_net,
            "avg_net_ret": prod_df["net_ret"].mean(),
            "sharpe": sharpe,
            "avg_turnover": prod_df["turnover"].mean(),
        })
        print(f"  α={alpha:.2f} | excess={ex_net:+.2%} | sharpe={sharpe:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(Path(__file__).parent / "exp_02_results.csv", index=False)
    print(f"\nResults saved to experiments/exp_02_results.csv")


if __name__ == "__main__":
    main()
