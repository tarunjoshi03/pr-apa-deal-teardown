"""
main.py
Full pipeline: production data → decline curve fit → type curve → IRR model → outputs

Run:
    python main.py

Outputs (./outputs/):
    well_production.csv      — synthetic production data (50 wells, 20 years)
    fitted_parameters.csv    — per-well Arps decline fit parameters
    type_curve.png           — P10/P50/P90 type curve plots
    irr_sensitivity.png      — IRR heatmap (WTI × D&C cost)
    well_dcf.png             — single-well DCF waterfall
    irr_sensitivity.csv      — IRR table (machine-readable)
    deal_metrics.csv         — implied acquisition metrics

Sources:
    PR 10-K FY2025, EDGAR Accession 0001658566-26-000035, filed Feb 26 2026
    PR Q1 2025 Earnings Release, May 7, 2025
    TGS/ComboCurve Delaware 2nd Bone Spring analysis, Dec 2025
"""

import os, sys
import pandas as pd

BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE, "data"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

from generate_well_data import generate_dataset
from decline_curve import fit_all_wells
from type_curve import compute_type_curves, plot_type_curves
from irr_model import (
    build_irr_sensitivity,
    print_sensitivity_table,
    compute_implied_metrics,
    plot_irr_heatmap,
    plot_single_well_dcf,
    TENK_COSTS,
    DEAL_TERMS,
)


def run():
    print("=" * 65)
    print("  PR / APA CORP — $608MM DELAWARE BASIN A&D TEARDOWN")
    print("  Full Pipeline: Production Data → Type Curve → IRR Model")
    print("=" * 65)

    # ── Step 1: Generate synthetic production data ─────────────────────────────
    print("\n[1/5] Generating synthetic well production data...")
    prod_path = os.path.join(OUT, "well_production.csv")
    prod_df   = generate_dataset(n_wells=50, output_path=prod_path)

    # ── Step 2: Fit decline curves ─────────────────────────────────────────────
    print("\n[2/5] Fitting Arps hyperbolic decline curves...")
    fit_df = fit_all_wells(prod_df)
    fit_path = os.path.join(OUT, "fitted_parameters.csv")
    fit_df.to_csv(fit_path, index=False)
    print(f"  Saved → {fit_path}")

    # ── Step 3: Build type curves ──────────────────────────────────────────────
    print("\n[3/5] Building P10/P50/P90 type curves...")
    curves = compute_type_curves(fit_df)
    plot_type_curves(curves, output_path=os.path.join(OUT, "type_curve.png"))

    # ── Step 4: IRR sensitivity model (P50 type curve) ────────────────────────
    print("\n[4/5] Running IRR sensitivity model (10-K cost structure)...")
    p50_profile = curves["P50"]["profile"]
    sens_df     = build_irr_sensitivity(p50_profile)
    print_sensitivity_table(sens_df)

    # Save IRR table
    irr_csv = os.path.join(OUT, "irr_sensitivity.csv")
    sens_df.map(lambda x: f"{x:.1%}").to_csv(irr_csv)
    print(f"\n  IRR table saved → {irr_csv}")

    # ── Step 5: Plots & deal metrics ───────────────────────────────────────────
    print("\n[5/5] Generating charts and deal metrics...")
    plot_irr_heatmap(sens_df,
                     output_path=os.path.join(OUT, "irr_sensitivity.png"))
    plot_single_well_dcf(p50_profile, dc_mm=7.75, wti=60,
                          output_path=os.path.join(OUT, "well_dcf.png"))

    metrics = compute_implied_metrics()
    metrics_df = pd.DataFrame([
        {"metric": k.replace("_", " ").title(), "value": v}
        for k, v in metrics.items()
    ])
    metrics_path = os.path.join(OUT, "deal_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\nImplied Acquisition Metrics:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['metric']:<35} {row['value']:>12,}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"\n  All outputs saved to: {OUT}/")
    print(f"  {'File':<30}  Description")
    print(f"  {'-'*55}")
    files = [
        ("well_production.csv",    "Synthetic production data (50 wells × 20 yrs)"),
        ("fitted_parameters.csv",  "Arps decline fit params per well"),
        ("type_curve.png",         "P10/P50/P90 type curve plots"),
        ("irr_sensitivity.png",    "IRR heatmap (WTI × D&C cost)"),
        ("well_dcf.png",           "Single-well DCF — $7.75MM D&C, $60 WTI"),
        ("irr_sensitivity.csv",    "IRR table (machine-readable)"),
        ("deal_metrics.csv",       "Implied acquisition metrics"),
    ]
    for fname, desc in files:
        exists = "✓" if os.path.exists(os.path.join(OUT, fname)) else "✗"
        print(f"  {exists} {fname:<30}  {desc}")

    p50_eur   = curves["P50"]["eur_mboe"]
    p50_qi    = curves["P50"]["params"]["qi"]
    irr_base  = sens_df.loc["$60/bbl WTI", "$7.75MM D&C"]
    irr_record= sens_df.loc["$60/bbl WTI", "$7.00MM D&C"]

    print(f"""
  Key Results (P50 type curve, 10-K cost structure):
  ─────────────────────────────────────────────────
  P50 IP30 (total BOE/d)       : {p50_qi:>8,.0f}
  P50 EUR (total MBOE)         : {p50_eur:>8,.0f}
  P50 EUR oil (MBOE @ 45% cut) : {p50_eur*0.45:>8,.0f}

  IRR @ $60 WTI / $7.75MM D&C  : {irr_base:>7.0%}   (announcement cost)
  IRR @ $60 WTI / $7.00MM D&C  : {irr_record:>7.0%}   (FY2025 record cost)
  Deal WTI breakeven           : $30/bbl  (per PR Q1 2025 ER)

  Cost Structure Source:
    PR 10-K FY2025, EDGAR Acc. 0001658566-26-000035
    LOE: $5.26/BOE | GP&T: $1.40/BOE | Sev: $2.72/BOE
    """)


if __name__ == "__main__":
    run()
