"""Experiment #21: Validate and optimize ensemble weights.

Question: Is the hardcoded 50/25/25 (XGB/Ridge/RF) optimal?
Hypothesis: Ridge may deserve more weight (heavy regularisation generalises better
             for noisy financial data). Equal weights may match or beat.

Variants:
  A) Baseline: 50/25/25 (current)
  B) Equal: 33/33/33
  C) Ridge-heavy: 25/50/25
  D) XGB-solo: 100/0/0
  E) Ridge-solo: 0/100/0
  F) RF-solo: 0/0/100
  G) Adaptive: OOS RC-based weighting (lookback=4, floor=10%)
  H) No-XGB: 0/50/50
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_experiments import (
    risk_model_df, feature_cols_all, close_prices,
    run_walk_forward, summarise, print_comparison,
    PROD_CFG,
)
from stock_data.config import ENS_W


def main():
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT #21: Ensemble Weight Validation")
    print("=" * 80)
    print(f"  Current weights: {ENS_W}")

    variants = [
        ("baseline_50/25/25", {"xgb": 0.50, "ridge": 0.25, "rf": 0.25}, False),
        ("equal_33/33/33", {"xgb": 1/3, "ridge": 1/3, "rf": 1/3}, False),
        ("ridge_heavy_25/50/25", {"xgb": 0.25, "ridge": 0.50, "rf": 0.25}, False),
        ("xgb_solo", {"xgb": 1.0, "ridge": 0.0, "rf": 0.0}, False),
        ("ridge_solo", {"xgb": 0.0, "ridge": 1.0, "rf": 0.0}, False),
        ("rf_solo", {"xgb": 0.0, "ridge": 0.0, "rf": 1.0}, False),
        ("no_xgb_0/50/50", {"xgb": 0.0, "ridge": 0.50, "rf": 0.50}, False),
        ("adaptive", None, True),
    ]

    results = []
    for label, weights, adaptive in variants:
        print(f"\n>>> {label}")
        kwargs = {"label": label}
        if adaptive:
            kwargs["adaptive_weights"] = True
            kwargs["adaptive_floor"] = 0.10
            kwargs["adaptive_lookback"] = 4
        else:
            kwargs["ens_weights"] = weights
        df = run_walk_forward(
            risk_model_df, feature_cols_all, close_prices, PROD_CFG,
            **kwargs,
        )
        results.append(df)

    # Per-model RC (same for all since we just change ensemble mixing)
    print("\n" + "=" * 80)
    print("PER-MODEL RANK CORRELATIONS (same across variants — model training unchanged)")
    print("=" * 80)
    bl = results[0]
    print(f"  XGB:   {bl['ret_rc_xgb'].mean():.4f}")
    print(f"  Ridge: {bl['ret_rc_ridge'].mean():.4f}")
    print(f"  RF:    {bl['ret_rc_rf'].mean():.4f}")

    # Portfolio comparison
    print_comparison(results)

    # Ablation: does ensemble add value over best single model?
    print("\n" + "=" * 80)
    print("ABLATION: Ensemble vs Best Solo Model")
    print("=" * 80)
    summaries = [(df["label"].iloc[0], summarise(df, df["label"].iloc[0])) for df in results]
    solos = [s for s in summaries if "solo" in s[0]]
    ensembles = [s for s in summaries if "solo" not in s[0]]

    best_solo = max(solos, key=lambda x: x[1]["ir_ann"])
    best_ens = max(ensembles, key=lambda x: x[1]["ir_ann"])
    print(f"  Best solo:     {best_solo[0]} (IR={best_solo[1]['ir_ann']:.3f}, IC={best_solo[1]['avg_ret_rc']:.4f})")
    print(f"  Best ensemble: {best_ens[0]} (IR={best_ens[1]['ir_ann']:.3f}, IC={best_ens[1]['avg_ret_rc']:.4f})")
    if best_ens[1]["ir_ann"] > best_solo[1]["ir_ann"]:
        print("  ✓ Ensemble adds value over best solo model.")
    else:
        print("  ⚠ Best solo matches or beats ensemble — ensemble may not be needed.")

    # Paired tests
    print("\n" + "=" * 80)
    print("PAIRED TESTS: Excess return vs baseline (50/25/25)")
    print("=" * 80)
    bl_excess = results[0]["net_ret"] - results[0]["mkt_ret"]
    for df in results[1:]:
        label = df["label"].iloc[0]
        ex = df["net_ret"] - df["mkt_ret"]
        diff = ex.values - bl_excess.values
        n = len(diff)
        mean_d = diff.mean()
        se_d = diff.std() / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean_d / se_d if se_d > 0 else 0
        print(f"  {label:25s}: Δ={mean_d:+.3%}, t={t_stat:.2f}")

    # Adaptive weight detail
    print("\n" + "=" * 80)
    print("ADAPTIVE WEIGHTS DETAIL (per quarter)")
    print("=" * 80)
    ada = results[-1]  # adaptive
    for _, row in ada.iterrows():
        print(f"  {row['test_date']}: w_xgb={row['ens_w_xgb']:.3f} w_ridge={row['ens_w_ridge']:.3f} "
              f"w_rf={row['ens_w_rf']:.3f} | IC_ens={row['ret_rc']:.4f}")

    # Final ranking
    print("\n" + "=" * 80)
    print("FINAL RANKING (by IR)")
    print("=" * 80)
    ranked = sorted(summaries, key=lambda x: x[1]["ir_ann"], reverse=True)
    for i, (label, s) in enumerate(ranked, 1):
        print(f"  {i}. {label:25s}: IR={s['ir_ann']:.3f}, excess={s['avg_excess_net']:+.3%}, "
              f"IC={s['avg_ret_rc']:.4f}, Sharpe={s['sharpe_ann']:.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save
    combined = pd.concat(results, ignore_index=True)
    out = Path(__file__).resolve().parent / "exp_21_results.csv"
    combined.to_csv(out, index=False)
    print(f"Results saved: {out}")


if __name__ == "__main__":
    main()
