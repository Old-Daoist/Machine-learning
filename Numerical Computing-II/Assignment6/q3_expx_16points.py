"""
Q3. Does More Data Always Mean Better Interpolation?
f(x) = e^x, 16 unequally spaced points on [-1, 1].
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from newton_interp import newton_coefficients, newton_eval_array


X_ALL = [-1.0, -0.92, -0.75, -0.61, -0.48, -0.31, -0.17, 0.02,
          0.15, 0.29, 0.43, 0.58, 0.67, 0.81, 0.93, 1.0]


def f(x):
    return math.exp(x)


def run():
    y_all = [f(xi) for xi in X_ALL]

    eval_pts = np.linspace(-1, 1, 100)
    exact = np.exp(eval_pts)

    subset_sizes = [4, 8, 12, 16]
    results = {}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(eval_pts, exact, color="black", lw=2.2, label="exact $f(x)=e^x$")

    fig2, ax2 = plt.subplots(figsize=(7.5, 5))

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

    for n_pts, col in zip(subset_sizes, colors):
        xs = X_ALL[:n_pts]
        ys = y_all[:n_pts]
        coeffs = newton_coefficients(xs, ys)
        P = np.array(newton_eval_array(xs, coeffs, eval_pts))
        err = np.abs(exact - P)
        emax = err.max()
        eloc = eval_pts[err.argmax()]
        results[n_pts] = (emax, eloc)

        ax.plot(eval_pts, P, ls="--", lw=1.3, color=col,
                label=f"$P_{{{n_pts-1}}}(x)$ ({n_pts} pts)")
        ax2.semilogy(eval_pts, err + 1e-18, color=col, label=f"{n_pts} points")

        print(f"n_points={n_pts:2d}  degree={n_pts-1:2d}  Emax={emax:.6e}  at x={eloc:+.4f}")

    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title("Q3: Newton interpolation of $e^x$ -- effect of number of points")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("q3_function_plot.png", dpi=150)
    plt.close(fig)

    ax2.set_xlabel("x"); ax2.set_ylabel("|e^x - P(x)|  (log scale)")
    ax2.set_title("Q3: Interpolation error vs number of points")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig("q3_error_plot.png", dpi=150)
    plt.close(fig2)

    print("\nSummary table:")
    print(f"{'points':>8}{'degree':>8}{'Emax':>16}{'location':>12}")
    for n_pts in subset_sizes:
        emax, eloc = results[n_pts]
        print(f"{n_pts:8d}{n_pts-1:8d}{emax:16.6e}{eloc:12.4f}")

    return results


if __name__ == "__main__":
    run()
