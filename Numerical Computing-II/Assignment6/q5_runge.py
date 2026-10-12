"""
Q5. Newton Interpolation and Numerical Error
f(x) = 1/(1+25x^2), -1<=x<=1, equally spaced points x_i = -1 + 2i/n, i=0..n
degrees n = 4, 8, 12, 16, 20 (i.e. 5,9,13,17,21 points)
"""
import numpy as np
import matplotlib.pyplot as plt
from newton_interp import newton_coefficients, newton_eval_array


def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def run():
    degrees = [4, 8, 12, 16, 20]
    eval_pts = np.linspace(-1, 1, 500)
    exact = f(eval_pts)

    results = {}
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(eval_pts, exact, color="black", lw=2.2, label="exact $f(x)=1/(1+25x^2)$")

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))  # error distribution, largest n

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]

    largest_err_curve = None

    for n, col in zip(degrees, colors):
        npts = n + 1
        x_nodes = [-1 + 2 * i / n for i in range(npts)]
        y_nodes = [f(xi) for xi in x_nodes]
        coeffs = newton_coefficients(x_nodes, y_nodes)
        P = np.array(newton_eval_array(x_nodes, coeffs, eval_pts))
        err = np.abs(exact - P)
        emax = err.max()
        results[n] = (npts, emax)

        ax1.plot(eval_pts, P, ls="--", lw=1.2, color=col,
                 label=f"$P_{{{n}}}(x)$ ({npts} pts)")

        if n == degrees[-1]:
            largest_err_curve = (eval_pts, err)

        print(f"n={n:3d}  points={npts:3d}  Emax={emax:.6e}")

    ax1.set_ylim(-1, 2)
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
    ax1.set_title("Q5: Newton interpolation of the Runge function (equally spaced nodes)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig("q5_function_plot.png", dpi=150)
    plt.close(fig1)

    # Plot 2: Emax vs n
    ns = degrees
    emaxs = [results[n][1] for n in ns]
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.semilogy(ns, emaxs, "o-", color="crimson")
    ax2.set_xlabel("polynomial degree n"); ax2.set_ylabel("Emax(n)  (log scale)")
    ax2.set_title("Q5: Maximum error vs polynomial degree (Runge phenomenon)")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig("q5_emax_vs_n.png", dpi=150)
    plt.close(fig2)

    # Plot 3: error distribution for largest n
    xs_e, err_e = largest_err_curve
    ax3.semilogy(xs_e, err_e + 1e-18, color="darkred")
    ax3.set_xlabel("x"); ax3.set_ylabel(f"|f(x)-P_{degrees[-1]}(x)|  (log)")
    ax3.set_title(f"Q5: Error distribution for n={degrees[-1]} (21 points)")
    ax3.grid(alpha=0.3, which="both")
    fig3.tight_layout(); fig3.savefig("q5_error_distribution.png", dpi=150)
    plt.close(fig3)

    print("\nSummary table:")
    print(f"{'n':>5}{'points':>10}{'Emax':>16}")
    for n in degrees:
        npts, emax = results[n]
        print(f"{n:5d}{npts:10d}{emax:16.6e}")

    return results


if __name__ == "__main__":
    run()
