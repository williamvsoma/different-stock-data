"""Experiment #34: Rank feature ablation.

Question: Are cross-sectional rank features redundant for tree models?
Hypothesis: Raw-only should match or beat all-features for XGB/RF (monotone transform = no info gain).
             Ranks-only should be better for Ridge (linearises nonlinear relationships).
             Model-specific (raw→trees, rank→Ridge) should be optimal.

Variants:
  A) Baseline: all features (raw + rank) for all models
  B) Raw only: raw features for all models
  C) Rank only: rank features for all models
  D) Model-specific: raw for XGB/RF, rank for Ridge
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_experiments import (
    risk_model_df, feature_cols_all, close_prices, raw_cols, rank_cols,
    run_walk_forward, summarise, print_comparison,
    experiment_baseline, experiment_34_model_specific_features,
    experiment_34_raw_only, experiment_34_rank_only,
    PROD_CFG,
)
from stock_data.config import ENS_W
import numpy as np
import pandas as pd


def main():
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT #34: Rank Feature Ablation")
    print("=" * 80)
    print(f"\nFeature counts:")
    print(f"  All features:  {len(feature_cols_all)}")
    print(f"  Raw only:      {len(raw_cols)}")
    print(f"  Rank only:     {len(rank_cols)}")
    print()

    # Run all 4 variants
    baseline = experiment_baseline()
    raw_only = experiment_34_raw_only()
    rank_only = experiment_34_rank_only()
    model_specific = experiment_34_model_specific_features()

    # Per-model rank correlation breakdown
    print("\n" + "=" * 80)
    print("PER-MODEL RANK CORRELATION (mean across quarters)")
    print("=" * 80)
    for label, df in [("baseline", baseline), ("raw_only", raw_only),
                      ("rank_only", rank_only), ("model_specific", model_specific)]:
        print(f"\n  {label:20s}: XGB={df['ret_rc_xgb'].mean():.4f}  "
              f"Ridge={df['ret_rc_ridge'].mean():.4f}  "
              f"RF={df['ret_rc_rf'].mean():.4f}  "
              f"Ens={df['ret_rc'].mean():.4f}")

    # Statistical tests: paired difference in ensemble RC
    print("\n" + "=" * 80)
    print("PAIRED TESTS: Excess net return difference vs baseline")
    print("=" * 80)
    bl_excess = baseline["net_ret"] - baseline["mkt_ret"]
    for label, df in [("raw_only", raw_only), ("rank_only", rank_only),
                      ("model_specific", model_specific)]:
        ex = df["net_ret"] - df["mkt_ret"]
        diff = ex.values - bl_excess.values
        n = len(diff)
        mean_d = diff.mean()
        se_d = diff.std() / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean_d / se_d if se_d > 0 else 0
        print(f"  {label:20s}: Δexcess={mean_d:+.3%}, SE={se_d:.3%}, t={t_stat:.2f} (N={n})")

    # Comparison table
    print_comparison([baseline, raw_only, rank_only, model_specific])

    # Decision
    print("\n" + "=" * 80)
    print("DECISION CRITERIA")
    print("=" * 80)
    s_bl = summarise(baseline, "baseline")
    s_raw = summarise(raw_only, "raw_only")
    s_rank = summarise(rank_only, "rank_only")
    s_ms = summarise(model_specific, "model_specific")

    results = [
        ("baseline", s_bl),
        ("raw_only", s_raw),
        ("rank_only", s_rank),
        ("model_specific", s_ms),
    ]
    best = max(results, key=lambda x: x[1]["ir_ann"])
    print(f"\n  Best by IR: {best[0]} (IR={best[1]['ir_ann']:.3f})")
    best_rc = max(results, key=lambda x: x[1]["avg_ret_rc"])
    print(f"  Best by IC: {best_rc[0]} (IC={best_rc[1]['avg_ret_rc']:.4f})")
    best_sharpe = max(results, key=lambda x: x[1]["sharpe_ann"])
    print(f"  Best by Sharpe: {best_sharpe[0]} (Sharpe={best_sharpe[1]['sharpe_ann']:.3f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    combined = pd.concat([baseline, raw_only, rank_only, model_specific], ignore_index=True)
    out = Path(__file__).resolve().parent / "exp_34_results.csv"
    combined.to_csv(out, index=False)
    print(f"Results saved: {out}")


if __name__ == "__main__":
    main()
