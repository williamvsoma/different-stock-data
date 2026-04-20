"""Experiment #32: Volatility model hyperparameters and benchmarks.

Question: Does the separate vol model (deeper trees + quality gate) outperform
          using same params as return model or naive historical vol?

Variants:
  A) Baseline (current master): XGB_VOL_PARAMS (depth=5) + quality gate (RC<0.10 → hist_vol_3m)
  B) Same-params: use XGB_PARAMS (depth=3) for vol — the old/original approach
  C) Naive only: always use hist_vol_3m (no ML vol model)
  D) No gate: XGB_VOL_PARAMS but disable quality gate (always trust ML)
  E) Aggressive gate: gate at RC < 0.20 (more fallback to hist)
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
from stock_data.config import XGB_PARAMS, XGB_VOL_PARAMS, VOL_FLOOR, VOL_RC_GATE


def main():
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT #32: Volatility Model Hyperparameters & Benchmarks")
    print("=" * 80)
    print(f"\n  XGB_PARAMS (return model): depth={XGB_PARAMS['max_depth']}, "
          f"min_child={XGB_PARAMS['min_child_weight']}, reg_lambda={XGB_PARAMS['reg_lambda']}")
    print(f"  XGB_VOL_PARAMS (vol model): depth={XGB_VOL_PARAMS['max_depth']}, "
          f"min_child={XGB_VOL_PARAMS['min_child_weight']}, reg_lambda={XGB_VOL_PARAMS['reg_lambda']}")
    print(f"  VOL_FLOOR={VOL_FLOOR}, VOL_RC_GATE={VOL_RC_GATE}")

    # A) Baseline: current master settings (deep vol + gate)
    print("\n>>> A) Baseline: XGB_VOL_PARAMS (depth=5) + quality gate (RC<0.10)")
    baseline = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=XGB_VOL_PARAMS,
        vol_rc_gate=VOL_RC_GATE,
        label="baseline_vol_deep+gate",
    )

    # B) Same params as return model (issue's original concern)
    print("\n>>> B) Same-params: XGB_PARAMS (depth=3) for vol + gate")
    same_params = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=XGB_PARAMS,
        vol_rc_gate=VOL_RC_GATE,
        label="vol_same_params+gate",
    )

    # C) Same params, no gate (original code before fixes)
    print("\n>>> C) Same-params, no gate (original code)")
    same_no_gate = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=XGB_PARAMS,
        vol_rc_gate=None,
        label="vol_same_no_gate",
    )

    # D) Deep vol but no gate
    print("\n>>> D) Deep vol (depth=5) but no quality gate")
    deep_no_gate = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=XGB_VOL_PARAMS,
        vol_rc_gate=None,
        label="vol_deep_no_gate",
    )

    # E) Aggressive gate (RC < 0.20)
    print("\n>>> E) Deep vol + aggressive gate (RC<0.20)")
    aggressive_gate = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=XGB_VOL_PARAMS,
        vol_rc_gate=0.20,
        label="vol_deep+gate_0.20",
    )

    # F) Very deep vol (depth=7)
    very_deep_params = XGB_VOL_PARAMS.copy()
    very_deep_params["max_depth"] = 7
    very_deep_params["min_child_weight"] = 3
    print("\n>>> F) Very deep vol (depth=7, min_child=3) + gate")
    very_deep = run_walk_forward(
        risk_model_df, feature_cols_all, close_prices, PROD_CFG,
        vol_params=very_deep_params,
        vol_rc_gate=VOL_RC_GATE,
        label="vol_very_deep+gate",
    )

    all_runs = [baseline, same_params, same_no_gate, deep_no_gate, aggressive_gate, very_deep]

    # Vol RC comparison (key metric for vol model quality)
    print("\n" + "=" * 80)
    print("VOL MODEL QUALITY (rank correlation with realised vol)")
    print("=" * 80)
    for df in all_runs:
        label = df["label"].iloc[0]
        vrc = df["vol_rc"].mean()
        rrc = df["ret_rc"].mean()
        print(f"  {label:30s}: vol_RC={vrc:.4f}  ret_RC={rrc:.4f}")

    # Full comparison
    print_comparison(all_runs)

    # Statistical comparison
    print("\n" + "=" * 80)
    print("PAIRED TESTS: Net excess return vs baseline")
    print("=" * 80)
    bl_excess = baseline["net_ret"] - baseline["mkt_ret"]
    for df in all_runs[1:]:
        label = df["label"].iloc[0]
        ex = df["net_ret"] - df["mkt_ret"]
        diff = ex.values - bl_excess.values
        n = len(diff)
        mean_d = diff.mean()
        se_d = diff.std() / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean_d / se_d if se_d > 0 else 0
        print(f"  {label:30s}: Δ={mean_d:+.3%}, t={t_stat:.2f}")

    # Decision
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    results = [(df["label"].iloc[0], summarise(df, df["label"].iloc[0])) for df in all_runs]
    best_vrc = max(results, key=lambda x: x[1]["avg_vol_rc"])
    best_ir = max(results, key=lambda x: x[1]["ir_ann"])
    print(f"  Best vol RC:  {best_vrc[0]} ({best_vrc[1]['avg_vol_rc']:.4f})")
    print(f"  Best IR:      {best_ir[0]} ({best_ir[1]['ir_ann']:.3f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save
    combined = pd.concat(all_runs, ignore_index=True)
    out = Path(__file__).resolve().parent / "exp_32_results.csv"
    combined.to_csv(out, index=False)
    print(f"Results saved: {out}")


if __name__ == "__main__":
    main()
