"""Plotting utilities for strategy diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_walk_forward_diagnostics(prod_df, factor_results, boot_means, ci_lo, ci_hi, ex_n):
    """Six-panel diagnostic plot from the walk-forward engine."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    # 1 — Cumulative wealth
    ax = axes[0, 0]
    wealth_prod = (1 + prod_df["net_ret"]).cumprod()
    wealth_mkt = (1 + prod_df["mkt_ret"]).cumprod()
    ax.plot(range(len(prod_df)), wealth_prod.values, "g-o", lw=2.5,
            label=f"Production MV net ({wealth_prod.iloc[-1]:.2f}x)")
    ax.plot(range(len(prod_df)), wealth_mkt.values, "k--o", lw=2,
            label=f"Market ({wealth_mkt.iloc[-1]:.2f}x)")

    fcolors = {"Low Vol": "blue", "Momentum (3m)": "orange",
               "Quality (ROE)": "purple", "Value (FCF/A)": "brown"}
    for fname, results in factor_results.items():
        if not results:
            continue
        fdf = pd.DataFrame(results)
        fm = fdf[fdf["test_date"].isin(prod_df["test_date"].values)].sort_values(
            "test_date").reset_index(drop=True)
        if len(fm) >= 2:
            fw = (1 + fm["port_ret"]).cumprod()
            ax.plot(range(len(fm)), fw.values, "-s", lw=1.2,
                    label=f"{fname} ({fw.iloc[-1]:.2f}x)",
                    color=fcolors.get(fname, "gray"), alpha=0.7)
    ax.set_xticks(range(len(prod_df)))
    ax.set_xticklabels([str(d.date()) for d in prod_df["test_date"]],
                        rotation=45, ha="right", fontsize=7)
    ax.set_title("Cumulative Wealth: Production MV vs Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # 2 — Per-quarter excess
    ax = axes[0, 1]
    colors = ["green" if e > 0 else "red" for e in ex_n]
    ax.bar(range(len(prod_df)), ex_n * 100, color=colors, alpha=0.7, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(ex_n.mean() * 100, color="blue", ls="--",
               label=f"Avg: {ex_n.mean():+.2%}")
    ax.set_xticks(range(len(prod_df)))
    ax.set_xticklabels([str(d.date()) for d in prod_df["test_date"]],
                        rotation=45, ha="right", fontsize=7)
    ax.set_title("Net Excess Return per Quarter (%)")
    ax.set_ylabel("Excess Return (%)")
    ax.legend()

    # 3 — Model quality
    ax = axes[1, 0]
    ax.plot(range(len(prod_df)), prod_df["ret_rc"], "b-o", lw=2, label="Return (ensemble)")
    ax.plot(range(len(prod_df)), prod_df["vol_rc"], "r-o", lw=2, label="Volatility")
    ax.plot(range(len(prod_df)), prod_df["ret_rc_xgb"], "b--", alpha=0.4, label="Ret (XGB)")
    ax.plot(range(len(prod_df)), prod_df["ret_rc_ridge"], "g--", alpha=0.4, label="Ret (Ridge)")
    ax.plot(range(len(prod_df)), prod_df["ret_rc_rf"], "m--", alpha=0.4, label="Ret (RF)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(prod_df)))
    ax.set_xticklabels([str(d.date()) for d in prod_df["test_date"]],
                        rotation=45, ha="right", fontsize=7)
    ax.set_title("Model Quality: Rank Correlation Over Time")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 4 — Concentration
    ax = axes[1, 1]
    ax.bar(range(len(prod_df)), prod_df["n_held"], color="steelblue", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(prod_df)), prod_df["max_wt"] * 100, "ro-", lw=1.5)
    ax.set_xticks(range(len(prod_df)))
    ax.set_xticklabels([str(d.date()) for d in prod_df["test_date"]],
                        rotation=45, ha="right", fontsize=7)
    ax.set_title("Portfolio Concentration")
    ax.set_ylabel("# Holdings (bars)")
    ax2.set_ylabel("Max Weight % (line)")
    ax.grid(alpha=0.3)

    # 5 — Turnover
    ax = axes[2, 0]
    ax.bar(range(len(prod_df)), prod_df["turnover"] * 100, color="orange", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(prod_df)), prod_df["tx_cost"] * 10000, "r-o", lw=1.5)
    ax.set_xticks(range(len(prod_df)))
    ax.set_xticklabels([str(d.date()) for d in prod_df["test_date"]],
                        rotation=45, ha="right", fontsize=7)
    ax.set_title("Turnover & Transaction Costs")
    ax.set_ylabel("One-Way Turnover % (bars)")
    ax2.set_ylabel("TX Cost (bps, line)")
    ax.grid(alpha=0.3)

    # 6 — Bootstrap
    ax = axes[2, 1]
    ax.hist(boot_means * 100, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=2, ls="--", label="Zero excess")
    ax.axvline(ex_n.mean() * 100, color="green", lw=2,
               label=f"Observed: {ex_n.mean():+.2%}")
    ax.axvline(ci_lo * 100, color="gray", lw=1, ls=":")
    ax.axvline(ci_hi * 100, color="gray", lw=1, ls=":",
               label=f"95% CI: [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    ax.set_title("Bootstrap: Excess Return Distribution")
    ax.set_xlabel("Quarterly Excess Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_simulation(sim_df, mkt_sim, qlog, initial_capital):
    """Four-panel simulation plot."""
    combined = sim_df[["portfolio_value"]].join(mkt_sim[["market_value"]], how="inner")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Portfolio value
    ax = axes[0, 0]
    ax.plot(combined.index, combined["portfolio_value"] / 1000,
            "g-", lw=2, label="Strategy")
    ax.plot(combined.index, combined["market_value"] / 1000,
            "k--", lw=1.5, label="Market (EW)")
    ax.set_title("Portfolio Value ($000s)")
    ax.set_ylabel("$000s")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # Drawdown
    ax = axes[0, 1]
    peak = combined["portfolio_value"].cummax()
    dd = (combined["portfolio_value"] / peak - 1) * 100
    peak_m = combined["market_value"].cummax()
    dd_m = (combined["market_value"] / peak_m - 1) * 100
    ax.fill_between(combined.index, dd, 0, alpha=0.4, color="red", label="Strategy")
    ax.fill_between(combined.index, dd_m, 0, alpha=0.3, color="gray", label="Market")
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # Quarterly returns
    if len(qlog) > 0:
        ax = axes[1, 0]
        x = range(len(qlog))
        colors = ["green" if r > m else "red"
                  for r, m in zip(qlog["sim_return"], qlog["market_return"])]
        ax.bar(x, qlog["sim_return"] * 100, color=colors, alpha=0.7, label="Strategy")
        ax.bar(x, qlog["market_return"] * 100, alpha=0.3, color="gray", label="Market")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(d.date()) for d in qlog["quarter"]], rotation=45, fontsize=8)
        ax.set_title("Quarterly Returns — Strategy vs Market")
        ax.set_ylabel("Return %")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend()
        ax.grid(alpha=0.3)

    # Rolling Sharpe
    ax = axes[1, 1]
    if "daily_return" in sim_df.columns and len(sim_df) > 63:
        rs = (sim_df["daily_return"].rolling(63).mean()
              / sim_df["daily_return"].rolling(63).std()) * np.sqrt(252)
        ax.plot(rs.index, rs.values, "g-", lw=1.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("Rolling 63-Day Annualized Sharpe")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return fig
