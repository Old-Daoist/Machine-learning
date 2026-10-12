"""
Q2. Newton Interpolation with Unequally Spaced Data
f(x) = sin(x), data at x = 0, 0.2, 0.55, 1.0, 1.4
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from newton_interp import (divided_difference_table, newton_coefficients,
                            newton_eval, newton_eval_array, print_dd_table)


def f(x):
    return math.sin(x)


def run():
    x = [0, 0.2, 0.55, 1.0, 1.4]
    y = [f(xi) for xi in x]

    print("=" * 60)
    print("Q2 -- Divided-difference table (unequally spaced)")
    print("=" * 60)
    print_dd_table(x, y)

    coeffs = newton_coefficients(x, y)
    print("\nNewton coefficients [a0..a4]:")
    for i, c in enumerate(coeffs):
        print(f"  a{i} = {c:.10g}")

    print("\nEvaluation at x = 0.4, 0.8, 1.2:")
    query = [0.4, 0.8, 1.2]
    print(f"{'x':>8} {'sin(x)':>12} {'P4(x)':>14} {'E(x)=|sinx-P4|':>16}")
    for xq in query:
        pv = newton_eval(x, coeffs, xq)
        fv = f(xq)
        print(f"{xq:8.3f} {fv:12.8f} {pv:14.10f} {abs(fv - pv):16.3e}")

    # Plots -------------------------------------------------------------
    xs_fine = np.linspace(0, 1.4, 400)
    exact = np.sin(xs_fine)
    interp = np.array(newton_eval_array(x, coeffs, xs_fine))
    err = np.abs(exact - interp)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_fine, exact, label="exact  $f(x)=\\sin x$", color="black", lw=2)
    ax.plot(xs_fine, interp, label="Newton interpolant $P_4(x)$", color="crimson",
            lw=1.5, ls="--")
    ax.scatter(x, y, color="blue", zorder=5, label="data points")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title("Q2: Newton interpolation of sin(x) (unequally spaced nodes)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("q2_function_plot.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(xs_fine, err, color="darkorange")
    ax2.set_xlabel("x"); ax2.set_ylabel("E(x) = |sin x - P4(x)|")
    ax2.set_title("Q2: Interpolation error over [0, 1.4]")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("q2_error_plot.png", dpi=150)
    plt.close(fig2)

    print(f"\nMax error over dense grid [0,1.4]: {err.max():.3e} at x={xs_fine[err.argmax()]:.4f}")
    return x, y, coeffs


if __name__ == "__main__":
    run()
