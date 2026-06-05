"""
irr_model.py
Computes well-level unlevered IRR from type curve + cost assumptions.

Cost structure sourced directly from:
  PR 10-K FY2025, EDGAR Accession 0001658566-26-000035
  Filed: February 26, 2026 | Item 7 MD&A — Operating costs per Boe table

Deal terms sourced from:
  PR Q1 2025 Earnings Release, May 7, 2025
  PR Q2 2025 8-K / Earnings Release, August 7, 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


# ════════════════════════════════════════════════════════════════════════════════
# 10-K COST ASSUMPTIONS  (all sourced to EDGAR filing)
# ════════════════════════════════════════════════════════════════════════════════

TENK_COSTS = {
    # Operating costs — PR 10-K FY2025, Item 7 MD&A
    "loe_per_boe":          5.26,    # Lease operating expenses, $/BOE
    "gpt_per_boe":          1.40,    # Gathering, processing & transport, $/BOE
    "severance_per_boe":    2.72,    # Severance & ad valorem taxes, $/BOE (FLAT, not % of rev)

    # Realized prices — PR 10-K FY2025, Item 7 avg sales price table (excl. hedges)
    "oil_realized_bbl":     64.06,   # $/bbl oil, FY2025 actual
    "ngl_realized_bbl":     18.41,   # $/bbl NGL, FY2025 actual
    "gas_realized_mcf":      0.63,   # $/Mcf gas, FY2025 actual (Waha weakness)

    # Deal disclosure — PR Q1 2025 Earnings Release
    "nri":                   0.83,   # Net revenue interest
    "oil_cut":               0.45,   # Oil fraction of total BOE
    "ngl_cut":               0.25,   # NGL fraction
    # Gas cut = 1 - oil_cut - ngl_cut = 0.30
}

DEAL_TERMS = {
    "purchase_price_mm":    608,
    "net_acres":          13320,
    "production_boe_d":   12000,
    "gross_locations":      100,
    "wti_breakeven":         30,
    "reinvestment_rate":   0.35,
    "close_date":    "June 16, 2025",
}


# ════════════════════════════════════════════════════════════════════════════════
# WELL-LEVEL DCF & IRR
# ════════════════════════════════════════════════════════════════════════════════

def compute_annual_cashflows(type_curve_profile, costs=None, wti_override=None):
    """
    Compute 20-year annual net cash flows from a monthly type curve profile.

    Parameters
    ----------
    type_curve_profile : np.array  shape (240,)  monthly BOE/d
    costs              : dict, defaults to TENK_COSTS
    wti_override       : float, if set overrides 10-K oil price proportionally

    Returns
    -------
    np.array of 20 annual net cash flows (dollars)
    """
    if costs is None:
        costs = TENK_COSTS

    # If WTI override is given, scale oil realized price proportionally
    oil_price = costs["oil_realized_bbl"]
    if wti_override is not None:
        # 10-K implied WTI = $64.06 / 0.902 realization ≈ $71
        implied_wti = costs["oil_realized_bbl"] / 0.902
        oil_price   = wti_override * (costs["oil_realized_bbl"] / implied_wti)

    nri       = costs["nri"]
    oil_cut   = costs["oil_cut"]
    ngl_cut   = costs["ngl_cut"]
    gas_cut   = 1.0 - oil_cut - ngl_cut

    annual_ncf = []
    for yr in range(20):
        monthly = type_curve_profile[yr*12 : (yr+1)*12]
        if len(monthly) == 0:
            break
        # Monthly BOE/d → annual BOE
        annual_boe = np.sum(monthly * 30.4)

        # Revenue (NRI-weighted)
        oil_rev = annual_boe * oil_cut * oil_price            * nri
        ngl_rev = annual_boe * ngl_cut * costs["ngl_realized_bbl"] * nri
        gas_rev = annual_boe * gas_cut * (costs["gas_realized_mcf"] / 6.0) * nri * 6.0

        gross   = oil_rev + ngl_rev + gas_rev

        # Operating costs (NRI-weighted)
        opex    = annual_boe * (costs["loe_per_boe"] +
                                costs["gpt_per_boe"] +
                                costs["severance_per_boe"]) * nri

        annual_ncf.append(gross - opex)

    return np.array(annual_ncf)


def compute_irr(capex_mm, annual_ncf, max_iter=200):
    """
    Compute unlevered IRR via bisection on NPV.

    Parameters
    ----------
    capex_mm   : float, D&C cost in $MM (Year 0 outflow)
    annual_ncf : np.array of annual cash flows (dollars)

    Returns
    -------
    float IRR (e.g. 0.25 = 25%)
    """
    capex = capex_mm * 1_000_000
    cfs   = np.concatenate([[-capex], annual_ncf])

    def npv(r):
        return sum(cf / (1+r)**t for t, cf in enumerate(cfs))

    if npv(0.001) < 0:
        return -0.99   # IRR below 0
    if npv(5.0) > 0:
        return 5.0     # IRR above 500%

    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_npv(capex_mm, annual_ncf, discount_rate=0.10):
    """Compute NPV at a given discount rate."""
    capex = capex_mm * 1_000_000
    cfs   = np.concatenate([[-capex], annual_ncf])
    return sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cfs))


# ════════════════════════════════════════════════════════════════════════════════
# SENSITIVITY TABLE
# ════════════════════════════════════════════════════════════════════════════════

def build_irr_sensitivity(type_curve_profile,
                           wti_range=None,
                           dc_range=None,
                           costs=None):
    """
    Build a WTI price × D&C cost IRR sensitivity table.

    Returns pd.DataFrame (WTI as index, D&C as columns), values as decimals.
    """
    if wti_range is None:
        wti_range = [45, 50, 55, 60, 65, 70, 75]
    if dc_range is None:
        dc_range  = [6.5, 7.0, 7.5, 7.75, 8.0, 8.5, 9.0]
    if costs is None:
        costs = TENK_COSTS

    rows = {}
    for wti in wti_range:
        ncf = compute_annual_cashflows(type_curve_profile,
                                       costs=costs,
                                       wti_override=wti)
        row = {}
        for dc in dc_range:
            row[f"${dc:.2f}MM D&C"] = compute_irr(dc, ncf)
        rows[f"${wti}/bbl WTI"] = row

    df = pd.DataFrame(rows).T
    return df


def print_sensitivity_table(df):
    """Pretty-print the IRR sensitivity table to console."""
    print("\n" + "="*70)
    print("WELL-LEVEL UNLEVERED IRR SENSITIVITY TABLE")
    print("Cost structure: PR 10-K FY2025 (EDGAR 0001658566-26-000035)")
    print("="*70)
    formatted = df.map(lambda x: f"{x:.0%}" if not np.isnan(x) else "N/A")
    print(formatted.to_string())
    print("\n★ $7.75MM = announcement D&C (~$775/ft, PR Q4 2024)")
    print("★ $7.00MM = FY2025 record D&C (~$700/ft, PR Q4 2025 ER, Feb 2026)")
    print("★ 20% IRR = typical A&D hurdle rate")


# ════════════════════════════════════════════════════════════════════════════════
# DEAL-LEVEL IMPLIED METRICS
# ════════════════════════════════════════════════════════════════════════════════

def compute_implied_metrics(deal=None):
    """Compute standard A&D acquisition metrics from deal terms."""
    if deal is None:
        deal = DEAL_TERMS
    pp = deal["purchase_price_mm"]
    return {
        "$/net_acre":              round(pp * 1e6 / deal["net_acres"]),
        "ev_per_flowing_boe_d":    round(pp * 1e6 / deal["production_boe_d"]),
        "ev_per_flowing_oil_bbl_d":round(pp * 1e6 / (deal["production_boe_d"] * TENK_COSTS["oil_cut"])),
        "ev_per_gross_location":   round(pp * 1e6 / deal["gross_locations"]),
        "annualized_boe_mboe":     round(deal["production_boe_d"] * 365 / 1000, 1),
        "ev_per_annual_boe":       round(pp * 1e3 / (deal["production_boe_d"] * 365 / 1000), 1),
    }


# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════

def plot_irr_heatmap(df, output_path="irr_sensitivity.png",
                     announcement_dc="$7.75MM D&C",
                     record_dc="$7.00MM D&C"):
    """
    Plot IRR sensitivity heatmap with A&D hurdle annotations.
    Color scheme: red (< 8%) → yellow (12%) → green (20%+)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    data = df.values.astype(float)
    n_rows, n_cols = data.shape

    # Custom colormap: red → yellow → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "irr_cmap",
        [(0.0, "#C0392B"), (0.12, "#F4D03F"), (0.30, "#2E8B57"), (1.0, "#1A7C3E")],
        N=256
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=0.50)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            txt = f"{val:.0%}" if not np.isnan(val) else "N/A"
            color = "white" if val >= 0.20 or val < 0.10 else "#1a1a1a"
            weight = "bold" if val >= 0.20 else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight=weight)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df.columns, fontsize=9, rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df.index, fontsize=9)

    # Highlight announcement and record D&C columns
    col_labels = list(df.columns)
    for dc_label, linestyle, label_text in [
        (announcement_dc, "--", "Announcement D&C"),
        (record_dc,       "-",  "FY2025 Record D&C"),
    ]:
        if dc_label in col_labels:
            col_idx = col_labels.index(dc_label)
            ax.axvline(x=col_idx - 0.5, color="#0D1B2A", lw=1.5, ls=linestyle, alpha=0.7)
            ax.axvline(x=col_idx + 0.5, color="#0D1B2A", lw=1.5, ls=linestyle, alpha=0.7)

    # 20% IRR hurdle annotation
    ax.text(n_cols + 0.1, n_rows / 2,
            "← 20% hurdle\n   rate",
            fontsize=8, color="#2E5F8A", va="center",
            transform=ax.transData)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Unlevered IRR", fontsize=9)
    cbar.set_ticks([0, 0.10, 0.20, 0.30, 0.40, 0.50])
    cbar.set_ticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])

    ax.set_title(
        "Well-Level Unlevered IRR Sensitivity\n"
        "PR / APA Corp. — $608MM Northern Delaware Basin Acquisition\n"
        "Cost structure: PR 10-K FY2025 (EDGAR Acc. 0001658566-26-000035, Item 7 MD&A)",
        fontsize=10, fontweight="bold", pad=12
    )
    ax.set_xlabel("D&C Cost per 2-Mile Lateral", fontsize=10)
    ax.set_ylabel("WTI Oil Price", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"IRR heatmap saved → {output_path}")


def plot_single_well_dcf(type_curve_profile, dc_mm=7.75, wti=60,
                          costs=None, output_path="well_dcf.png"):
    """Plot single well cumulative NCF and annual FCF bar chart."""
    if costs is None:
        costs = TENK_COSTS

    ncf      = compute_annual_cashflows(type_curve_profile, costs=costs,
                                        wti_override=wti)
    capex    = dc_mm * 1e6
    cum_ncf  = np.cumsum(np.concatenate([[-capex], ncf])) / 1e6
    irr      = compute_irr(dc_mm, ncf)
    npv10    = compute_npv(dc_mm, ncf, 0.10) / 1e6

    years = np.arange(0, len(cum_ncf))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    # Cumulative NCF
    ax1 = axes[0]
    ax1.set_facecolor("white")
    ax1.plot(years, cum_ncf, color="#2E5F8A", lw=2.5)
    ax1.axhline(0, color="#888", lw=0.8, ls="--")
    ax1.fill_between(years, cum_ncf, 0,
                     where=cum_ncf >= 0, alpha=0.12, color="#1A7C3E")
    ax1.fill_between(years, cum_ncf, 0,
                     where=cum_ncf < 0, alpha=0.12, color="#C0392B")
    payout_yr = next((y for y, v in zip(years, cum_ncf) if v >= 0), None)
    if payout_yr:
        ax1.axvline(payout_yr, color="#C9A84C", lw=1.5, ls=":", alpha=0.8)
        ax1.text(payout_yr + 0.2, cum_ncf.min() * 0.6,
                 f"Payout\nYr {payout_yr}", fontsize=8, color="#C9A84C")
    ax1.set_title(f"Cumulative NCF — ${dc_mm:.2f}MM D&C, ${wti}/bbl WTI\n"
                  f"IRR: {irr:.0%}  |  NPV10: ${npv10:.2f}MM",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("Cumulative NCF ($MM)", fontsize=10)
    ax1.grid(True, alpha=0.12)
    ax1.spines[["top","right"]].set_visible(False)

    # Annual FCF bar
    ax2 = axes[1]
    ax2.set_facecolor("white")
    bar_colors = ["#1A7C3E" if v >= 0 else "#C0392B" for v in ncf]
    ax2.bar(range(1, len(ncf)+1), ncf/1e6, color=bar_colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_title("Annual net cash flow per well ($MM)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("NCF ($MM)", fontsize=10)
    ax2.grid(True, alpha=0.12, axis="y")
    ax2.spines[["top","right"]].set_visible(False)

    plt.suptitle(
        "Single-well DCF  |  10-K cost structure  |  "
        "PR 10-K FY2025 (EDGAR 0001658566-26-000035)",
        fontsize=9, color="#555555", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"DCF plot saved → {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from generate_well_data import generate_dataset
    from decline_curve import fit_all_wells
    from type_curve import compute_type_curves, plot_type_curves

    print("Running IRR model standalone test...")
    prod_df = generate_dataset(n_wells=50)
    fit_df  = fit_all_wells(prod_df)
    curves  = compute_type_curves(fit_df)

    p50_profile = curves["P50"]["profile"]

    sens_df = build_irr_sensitivity(p50_profile)
    print_sensitivity_table(sens_df)

    metrics = compute_implied_metrics()
    print("\nImplied Acquisition Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<35} {v:>10,}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_irr_heatmap(sens_df,
                     output_path=os.path.join(out_dir, "irr_sensitivity.png"))
    plot_single_well_dcf(p50_profile, dc_mm=7.75, wti=60,
                          output_path=os.path.join(out_dir, "well_dcf.png"))
    plot_type_curves(curves,
                     output_path=os.path.join(out_dir, "type_curve.png"))
