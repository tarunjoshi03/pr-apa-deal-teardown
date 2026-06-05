"""
type_curve.py
Aggregates individual well decline fits into P10/P50/P90 type curves
and computes EUR distributions.

Methodology:
  1. Filter converged wells with R² > 0.85
  2. Compute percentile type curves from parameter distributions
  3. Generate 20-year monthly production profiles for P10/P50/P90
  4. Plot type curves with confidence band

Sources:
  - TGS/ComboCurve Delaware 2nd Bone Spring analysis (Dec 2025)
    P50 EUR ~78 bbl/ft oil ≈ 315 MBOE oil on 2-mi lateral → ~700 MBOE total
  - PR 10-K FY2025 (EDGAR 0001658566-26-000035)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from decline_curve import arps_with_terminal


# ── Type curve computation ─────────────────────────────────────────────────────

def compute_type_curves(fit_df, dt_annual=0.08, n_months=240):
    """
    Build P10/P50/P90 type curves from fitted well parameters.

    P10 = high case (top 10% producers)
    P50 = median
    P90 = low case

    Returns
    -------
    dict with keys 'P10', 'P50', 'P90', each a dict of:
        params : {qi, di_annual, b}
        profile: np.array of monthly BOE/d (length n_months)
        eur    : float EUR in MBOE
    """
    # Filter reliable fits
    good = fit_df[fit_df["converged"] & (fit_df["r_squared"] >= 0.85)].copy()
    if len(good) < 5:
        raise ValueError(f"Too few converged wells ({len(good)}) to build type curve.")

    t = np.arange(1, n_months + 1)
    curves = {}

    for label, pct in [("P10", 90), ("P50", 50), ("P90", 10)]:
        qi = np.percentile(good["qi"],        pct)
        di = np.percentile(good["di_annual"], 100 - pct)   # higher Di → lower EUR
        b  = min(np.percentile(good["b"],     pct), 1.40)  # cap b at 1.40 per Delaware norms

        profile = arps_with_terminal(t, qi, di, b, dt_annual)
        eur     = np.sum(profile * 30.4) / 1000.0   # MBOE

        curves[label] = {
            "params":  {"qi": round(qi,1), "di_annual": round(di,4), "b": round(b,3)},
            "profile": profile,
            "eur_mboe": round(eur, 1),
        }

    # Summary
    print("\nType Curve Summary (Delaware Bone Spring, 2-mi lateral)")
    print(f"  Based on {len(good)} wells with R² ≥ 0.85")
    print(f"  {'Case':<6}  {'IP30':>8}  {'Di':>8}  {'b':>6}  {'EUR (MBOE)':>12}")
    print(f"  {'-'*46}")
    for lbl, data in curves.items():
        p = data["params"]
        print(f"  {lbl:<6}  {p['qi']:>8.0f}  {p['di_annual']:>8.3f}  "
              f"{p['b']:>6.2f}  {data['eur_mboe']:>12.0f}")

    return curves


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_type_curves(curves, output_path="type_curve.png"):
    """
    Plot P10/P50/P90 type curves on a semi-log scale.
    Style: clean, publication-ready, no chart junk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    colors = {"P10": "#1A7C3E", "P50": "#2E5F8A", "P90": "#C9A84C"}
    t_months = np.arange(1, 241)
    t_years  = t_months / 12.0

    # ── Left: Semi-log rate plot ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("white")

    for label, data in curves.items():
        oil_profile = data["profile"] * 0.45   # oil cut
        ax1.semilogy(t_years, oil_profile, color=colors[label],
                     lw=2.0 if label == "P50" else 1.2,
                     ls="-" if label == "P50" else "--",
                     label=f"{label}  |  IP30: {data['params']['qi']*0.45:.0f} bbl/d  "
                           f"|  EUR: {data['eur_mboe']*0.45:.0f} MBOE oil")

    # Shade P10-P90 band
    ax1.fill_between(t_years,
                     curves["P90"]["profile"] * 0.45,
                     curves["P10"]["profile"] * 0.45,
                     alpha=0.08, color="#2E5F8A")

    ax1.set_xlabel("Time (years)", fontsize=11)
    ax1.set_ylabel("Oil rate (bbl/d)", fontsize=11)
    ax1.set_title("Type curve — oil rate  (2-mi lateral, Bone Spring Delaware)",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax1.set_xlim(0, 20)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, which="both", alpha=0.15)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Right: Cumulative EUR ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    for label, data in curves.items():
        cum_oil = np.cumsum(data["profile"] * 0.45 * 30.4) / 1000.0
        ax2.plot(t_years, cum_oil, color=colors[label],
                 lw=2.0 if label == "P50" else 1.2,
                 ls="-" if label == "P50" else "--",
                 label=f"{label}  |  20-yr EUR: {cum_oil[-1]:.0f} MBOE oil")

    ax2.fill_between(t_years,
                     np.cumsum(curves["P90"]["profile"] * 0.45 * 30.4) / 1000,
                     np.cumsum(curves["P10"]["profile"] * 0.45 * 30.4) / 1000,
                     alpha=0.08, color="#2E5F8A")

    ax2.set_xlabel("Time (years)", fontsize=11)
    ax2.set_ylabel("Cumulative oil production (MBOE)", fontsize=11)
    ax2.set_title("Cumulative EUR by case",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.set_xlim(0, 20)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.grid(True, alpha=0.15)
    ax2.spines[["top","right"]].set_visible(False)

    # Annotate P50 EUR on cumulative plot
    p50_cum = np.cumsum(curves["P50"]["profile"] * 0.45 * 30.4) / 1000
    ax2.annotate(f"P50 EUR\n{p50_cum[-1]:.0f} MBOE",
                 xy=(20, p50_cum[-1]),
                 xytext=(-60, -20), textcoords="offset points",
                 fontsize=9, color=colors["P50"],
                 arrowprops=dict(arrowstyle="->", color=colors["P50"], lw=1))

    plt.suptitle(
        "Delaware Basin — 2nd/3rd Bone Spring Type Curve\n"
        "Source: Synthetic production data calibrated to TGS/ComboCurve 2025 benchmarks; "
        "PR 10-K FY2025 (EDGAR 0001658566-26-000035)",
        fontsize=9, color="#555555", y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nType curve plot saved → {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from generate_well_data import generate_dataset
    from decline_curve import fit_all_wells

    prod_df = generate_dataset(n_wells=50)
    fit_df  = fit_all_wells(prod_df)
    curves  = compute_type_curves(fit_df)

    out_plot = os.path.join(os.path.dirname(__file__), "..", "outputs", "type_curve.png")
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plot_type_curves(curves, output_path=out_plot)
