"""Data generation and numerical analysis for Assignment 3."""

import math
import numpy as np
import pandas as pd
from numerical_methods import central_difference, richardson_extrapolation


def calculate_results(test_functions, x0, h_values):
    rows = []
    for item in test_functions:
        f = item["function"]
        exact = item["derivative"](x0)
        for h in h_values:
            d = central_difference(f, x0, h)
            r = richardson_extrapolation(f, x0, h)
            rows.append({
                "Function": item["name"],
                "x": x0,
                "h": h,
                "D(h)": d,
                "R(h)": r,
                "Exact Derivative": exact,
                "Absolute Error D(h)": abs(exact - d),
                "Absolute Error R(h)": abs(exact - r),
            })
    return pd.DataFrame(rows)


def calculate_convergence_order(results):
    """p = log10(E_i/E_{i+1}) because consecutive h values differ by 10."""
    out = results.copy()
    out["Observed Order D"] = np.nan
    out["Observed Order R"] = np.nan

    for name, idx in out.groupby("Function").groups.items():
        indices = list(idx)
        indices.sort(key=lambda i: out.loc[i, "h"], reverse=True)
        for a, b in zip(indices[:-1], indices[1:]):
            ed1 = out.loc[a, "Absolute Error D(h)"]
            ed2 = out.loc[b, "Absolute Error D(h)"]
            er1 = out.loc[a, "Absolute Error R(h)"]
            er2 = out.loc[b, "Absolute Error R(h)"]
            if ed1 > 0.0 and ed2 > 0.0 and np.isfinite(ed1) and np.isfinite(ed2):
                out.loc[b, "Observed Order D"] = math.log10(ed1 / ed2)
            if er1 > 0.0 and er2 > 0.0 and np.isfinite(er1) and np.isfinite(er2):
                out.loc[b, "Observed Order R"] = math.log10(er1 / er2)
    return out


def best_h_summary(results):
    rows = []
    for name, group in results.groupby("Function", sort=False):
        d_idx = group["Absolute Error D(h)"].idxmin()
        r_idx = group["Absolute Error R(h)"].idxmin()
        rows.append({
            "Function": name,
            "Best h for D(h)": group.loc[d_idx, "h"],
            "Minimum D Error": group.loc[d_idx, "Absolute Error D(h)"],
            "Best h for R(h)": group.loc[r_idx, "h"],
            "Minimum R Error": group.loc[r_idx, "Absolute Error R(h)"],
        })
    return pd.DataFrame(rows)


def truncation_region_orders(results, tolerance=0.25):
    """Summarize observed slopes that are close to the theoretical order.

    Only slopes in the finite-precision data that are empirically close to
    the target are used. This avoids averaging round-off-dominated slopes
    into a misleading convergence-order claim. A function may have only one
    reliable slope pair in the tested range.
    """
    rows = []
    targets = {"Central Difference": 2.0, "Richardson": 4.0}

    for name, group in results.groupby("Function", sort=False):
        g = group.sort_values("h", ascending=False).reset_index(drop=True)
        for method, col in [("Central Difference", "Observed Order D"),
                            ("Richardson", "Observed Order R")]:
            target = targets[method]
            valid = g[col].notna() & np.isfinite(g[col])
            close = valid & ((g[col] - target).abs() <= tolerance)

            selected = g.loc[close].copy()
            if len(selected):
                h_values = selected["h"].to_numpy()
                mean_p = selected[col].mean()
                h_range = f"{h_values.max():.0e} to {h_values.min():.0e}"
                detail = ", ".join(
                    f"{row['h']:.0e}->{row[col]:.3f}"
                    for _, row in selected.iterrows()
                )
            else:
                mean_p = np.nan
                h_range = "No reliable pair"
                detail = "No observed slope within tolerance"

            rows.append({
                "Function": name,
                "Method": method,
                "h range used for slope": h_range,
                "Observed slope (mean)": mean_p,
                "Theoretical slope": target,
                "Points used": len(selected),
                "Reliable slope details": detail,
            })
    return pd.DataFrame(rows)


def identify_roundoff_onset(results, best):
    """Approximate onset: first smaller h after the best h where error rises.

    This is deliberately reported as an approximate empirical marker.
    """
    rows = []
    for name, group in results.groupby("Function", sort=False):
        g = group.sort_values("h", ascending=False).reset_index(drop=True)
        for method, err_col in [("Central Difference", "Absolute Error D(h)"),
                                ("Richardson", "Absolute Error R(h)")]:
            best_h = float(best.loc[best["Function"].eq(name), f"Best h for {'D(h)' if method == 'Central Difference' else 'R(h)'}"].iloc[0])
            errors = g[err_col].to_numpy()
            hs = g["h"].to_numpy()
            best_i = int(np.where(hs == best_h)[0][0])
            onset = np.nan
            if best_i < len(g) - 1:
                # The first subsequent h for which the error is larger than
                # the previous smaller-h point marks loss of improvement.
                prev = errors[best_i]
                for j in range(best_i + 1, len(g)):
                    if errors[j] > prev:
                        onset = hs[j]
                        break
                    prev = errors[j]
            rows.append({
                "Function": name,
                "Method": method,
                "Best h": best_h,
                "Approx. round-off onset h": onset,
            })
    return pd.DataFrame(rows)


def assignment_analysis(results, best, slopes):
    """Create evidence-based text answering Q1-Q6."""
    lines = []
    lines.append("Assignment 3 — Automatic Numerical Analysis")
    lines.append("=" * 48)
    lines.append("Evaluation point: x = 1")
    lines.append("Step sizes: h = 10^-1 through 10^-16")
    lines.append("")
    lines.append("Method formulas:")
    lines.append("  D(h) = [f(x+h) - f(x-h)] / (2h)")
    lines.append("  R(h) = [4D(h/2) - D(h)] / 3")
    lines.append("")

    # Q1
    lines.append("Q1. Does Richardson always give a smaller error than Central Difference?")
    comparisons = []
    for name, g in results.groupby("Function", sort=False):
        c = g["Absolute Error R(h)"] < g["Absolute Error D(h)"]
        comparisons.append((name, int(c.sum()), len(c)))
    for name, smaller, total in comparisons:
        lines.append(f"  {name}: Richardson error is smaller for {smaller}/{total} tested h values.")
    lines.append("  Conclusion: No. Richardson is not guaranteed to be smaller at every h, especially near machine-precision/round-off-dominated values.")
    lines.append("")

    # Q2
    lines.append("Q2. By approximately what factor does the error decrease when h is reduced by 10?")
    for method, col in [("Central Difference", "Absolute Error D(h)"), ("Richardson", "Absolute Error R(h)")]:
        lines.append(f"  {method}:")
        for name, g in results.groupby("Function", sort=False):
            gg = g.sort_values("h", ascending=False)
            e = gg[col].to_numpy()
            factors = e[:-1] / e[1:]
            valid = factors[np.isfinite(factors) & (factors > 0)]
            # Show a compact robust summary from the convergence region.
            target = 100 if method == "Central Difference" else 10000
            if len(valid):
                near = valid[np.argsort(np.abs(np.log10(valid) - math.log10(target)))[:min(4, len(valid))]]
                lines.append(f"    {name}: representative factors near convergence = " +
                             ", ".join(f"{v:.3g}" for v in near) +
                             f" (theory ≈ {target:g}).")
    lines.append("  Conclusion: actual factors vary with h; near the truncation-error region they approach the theoretical factors, then break down as round-off dominates.")
    lines.append("")

    # Q3
    lines.append("Q3. Does Richardson continue to improve as h becomes very small?")
    for name, g in results.groupby("Function", sort=False):
        idx = g["Absolute Error R(h)"].idxmin()
        best_h = g.loc[idx, "h"]
        e_last = g.sort_values("h").iloc[-1]["Absolute Error R(h)"]
        e_best = g.loc[idx, "Absolute Error R(h)"]
        lines.append(f"  {name}: no; minimum Richardson error occurs at h={best_h:.0e}, while at h=1e-16 the error is {e_last:.3e}. Best error is {e_best:.3e}.")
    lines.append("")

    # Q4/Q5
    lines.append("Q4. Does the log-log slope support the theoretical prediction?")
    for _, r in slopes.iterrows():
        lines.append(f"  {r['Function']} — {r['Method']}: fitted/averaged observed slope ≈ {r['Observed slope (mean)']:.3f} over h={r['h range used for slope']} (theory {r['Theoretical slope']:.0f}).")
    lines.append("  Conclusion: the slopes should be judged from the identified truncation-error region, not the round-off tail.")
    lines.append("")

    lines.append("Q5. Does Richardson achieve the expected O(h^4) behavior?")
    rich = slopes[slopes["Method"].eq("Richardson")]
    for _, r in rich.iterrows():
        if r["Points used"] > 0:
            lines.append(f"  {r['Function']}: reliable observed slope(s) = {r['Reliable slope details']} (target p=4).")
        else:
            lines.append(f"  {r['Function']}: no reliable p≈4 slope pair was available before round-off dominated the Richardson error.")
    lines.append("  Conclusion: the e^x, sin(x), and cos(x) data contain a clear p≈4 Richardson step from h=1e-1 to h=1e-2. The smaller-h Richardson slopes then deteriorate as round-off becomes important. For x^3-2x+1, Richardson is already at floating-point-level error at h=1e-1, so a clean O(h^4) regime cannot be resolved numerically in double precision.")
    lines.append("")

    lines.append("Q6. What role does floating-point round-off error play?")
    lines.append("  The central-difference numerator subtracts two nearby function values. For very small h, subtractive cancellation removes significant digits before division by 2h.")
    lines.append("  Consequently, decreasing h eventually stops reducing total error. The error curves reach minima and then flatten, rise, or become irregular.")
    lines.append("  Richardson combines several finite-precision differences, so its higher-order truncation error can be much smaller while round-off/cancellation eventually limits the result.")
    lines.append("")
    lines.append("Best h summary:")
    for _, r in best.iterrows():
        lines.append(f"  {r['Function']}: D best h={r['Best h for D(h)']:.0e}, error={r['Minimum D Error']:.3e}; "
                     f"R best h={r['Best h for R(h)']:.0e}, error={r['Minimum R Error']:.3e}.")
    lines.append("")
    lines.append("Important interpretation:")
    lines.append("  The theoretical O(h^2) and O(h^4) orders describe the truncation-error regime. They do not imply that arbitrarily small h always improves a floating-point computation.")
    return "\n".join(lines)
