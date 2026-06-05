"""
decline_curve.py
Fits Arps hyperbolic decline parameters (qi, Di, b) to monthly production data
using non-linear least squares (scipy.optimize.curve_fit).

Arps hyperbolic decline:
    q(t) = qi / (1 + b * Di * t)^(1/b)

With exponential terminal switch at terminal decline rate Dt.

References:
    Arps, J.J. (1945). Analysis of Decline Curves. Trans. AIME, 160, 228-247.
    PR 10-K FY2025, EDGAR Acc. 0001658566-26-000035
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import warnings


# ── Decline model ──────────────────────────────────────────────────────────────

def hyperbolic_q(t, qi, di, b):
    """
    Pure hyperbolic rate at time t (months).
    No terminal switch — used for curve_fit only.
    """
    return qi / (1.0 + b * di * t) ** (1.0 / b)


def arps_with_terminal(t_months, qi, di_annual, b, dt_annual=0.08):
    """
    Hyperbolic decline with exponential terminal switch.
    Returns rate array (BOE/d).
    """
    di = di_annual / 12.0
    dt = dt_annual / 12.0
    q  = np.zeros_like(t_months, dtype=float)
    t_switch = (di / dt - 1) / (b * dt) if (b > 0 and di > dt) else np.inf
    for i, t in enumerate(t_months):
        if t <= t_switch:
            q[i] = qi / (1.0 + b * di * t) ** (1.0 / b)
        else:
            q_sw = qi / (1.0 + b * di * t_switch) ** (1.0 / b)
            q[i] = q_sw * np.exp(-dt * (t - t_switch))
    return q


# ── Per-well fitting ───────────────────────────────────────────────────────────

def fit_well(months, rates, bounds=None):
    """
    Fit hyperbolic decline to a single well's production history.

    Parameters
    ----------
    months : array-like, shape (N,)
        Month index (1-based).
    rates  : array-like, shape (N,)
        Observed BOE/d rates.

    Returns
    -------
    dict with keys: qi, di, b, r_squared, eur_mboe, converged
    """
    months = np.asarray(months, dtype=float)
    rates  = np.asarray(rates,  dtype=float)

    # Filter out zero/negative rates
    mask   = rates > 0
    t, q   = months[mask], rates[mask]
    if len(t) < 6:
        return _failed_fit()

    # Initial guess: qi = first observed rate, di = 0.06/month, b = 1.3
    p0 = [q[0], 0.06, 1.3]
    if bounds is None:
        bounds = ([0, 0.001, 0.3], [q[0]*2, 0.25, 2.0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(hyperbolic_q, t, q, p0=p0,
                                bounds=bounds, maxfev=5000)
        qi_fit, di_fit, b_fit = popt

        # Convert monthly Di to annual
        di_annual = di_fit * 12.0

        # R-squared
        q_pred = hyperbolic_q(t, qi_fit, di_fit, b_fit)
        ss_res = np.sum((q - q_pred) ** 2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # EUR: integrate out to 240 months
        t_full = np.arange(1, 241)
        q_full = arps_with_terminal(t_full, qi_fit, di_annual, b_fit)
        eur    = np.sum(q_full * 30.4) / 1000.0   # convert BOE/d·days → MBOE

        return {
            "qi":          round(float(qi_fit), 1),
            "di_annual":   round(float(di_annual), 4),
            "b":           round(float(b_fit), 3),
            "r_squared":   round(float(r2), 4),
            "eur_mboe":    round(float(eur), 1),
            "converged":   True,
        }
    except (RuntimeError, ValueError):
        return _failed_fit()


def _failed_fit():
    return {"qi": np.nan, "di_annual": np.nan, "b": np.nan,
            "r_squared": np.nan, "eur_mboe": np.nan, "converged": False}


# ── Batch fitting ──────────────────────────────────────────────────────────────

def fit_all_wells(df, well_col="well_id", month_col="month", rate_col="boe_d"):
    """
    Fit decline curves for all wells in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame with columns [well_col, month_col, rate_col]

    Returns
    -------
    pd.DataFrame with one row per well containing fitted parameters.
    """
    results = []
    for well_id, grp in df.groupby(well_col):
        grp = grp.sort_values(month_col)
        params = fit_well(grp[month_col].values, grp[rate_col].values)
        params["well_id"] = well_id
        results.append(params)

    fit_df = pd.DataFrame(results)
    fit_df = fit_df[["well_id", "qi", "di_annual", "b", "r_squared", "eur_mboe", "converged"]]

    n_total     = len(fit_df)
    n_converged = fit_df["converged"].sum()
    print(f"Fitted {n_converged}/{n_total} wells successfully "
          f"({100*n_converged/n_total:.0f}%)")
    print(f"  Median qi:       {fit_df['qi'].median():.0f} BOE/d")
    print(f"  Median Di:       {fit_df['di_annual'].median():.3f} /yr")
    print(f"  Median b:        {fit_df['b'].median():.2f}")
    print(f"  Median EUR:      {fit_df['eur_mboe'].median():.0f} MBOE")
    return fit_df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from generate_well_data import generate_dataset

    prod_df = generate_dataset(n_wells=50)
    fit_df  = fit_all_wells(prod_df)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "fitted_parameters.csv")
    fit_df.to_csv(out, index=False)
    print(f"\nSaved fitted parameters → {out}")
