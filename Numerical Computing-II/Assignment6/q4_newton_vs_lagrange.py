"""
Q4. Newton vs Lagrange -- Are They Really Different?
f(x) = 1/(1+x^2), 17 equally spaced points on [-4, 4]
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from newton_interp import newton_coefficients, newton_eval_array, lagrange_eval_array


def f(x):
    return 1.0 / (1.0 + x**2)


def run():
    x = [-4.0 + 0.5 * i for i in range(17)]
    y = [f(xi) for xi in x]

    eval_pts = np.linspace(-4, 4, 200)
    exact = 1.0 / (1.0 + eval_pts**2)

    coeffs = newton_coefficients(x, y)

    t0 = time.perf_counter()
    P_newton = np.array(newton_eval_array(x, coeffs, eval_pts))
    t_newton = time.perf_counter() - t0

    t0 = time.perf_counter()
    P_lagrange = np.array(lagrange_eval_array(x, y, eval_pts))
    t_lagrange = time.perf_counter() - t0

    E_N = np.abs(exact - P_newton)
    E_L = np.abs(exact - P_lagrange)

    print(f"max E_N = {E_N.max():.6e}  at x={eval_pts[E_N.argmax()]:+.3f}")
    print(f"max E_L = {E_L.max():.6e}  at x={eval_pts[E_L.argmax()]:+.3f}")
    print(f"max |P_newton - P_lagrange| (should be ~machine eps-level) = "
          f"{np.abs(P_newton-P_lagrange).max():.3e}")

    print(f"\nTiming for {len(eval_pts)} evaluation points:")
    print(f"  Newton  : {t_newton*1e3:.3f} ms (coeffs computed once, O(n) per point)")
    print(f"  Lagrange: {t_lagrange*1e3:.3f} ms (O(n^2) per point, no reusable coeffs)")

    query = [-3.75, -2.25, -0.75, 0.75, 2.25, 3.75]
    print(f"\n{'x':>8}{'f(x)':>12}{'P_N(x)':>14}{'P_L(x)':>14}{'E_N':>12}{'E_L':>12}")
    for xq in query:
        pn = newton_eval_array(x, coeffs, [xq])[0]
        pl = lagrange_eval_array(x, y, [xq])[0]
        fv = f(xq)
        print(f"{xq:8.2f}{fv:12.6f}{pn:14.6f}{pl:14.6f}{abs(fv-pn):12.3e}{abs(fv-pl):12.3e}")

    # Plot 1: exact + Lagrange + Newton
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(eval_pts, exact, color="black", lw=2, label="exact $f(x)=1/(1+x^2)$")
    ax.plot(eval_pts, P_newton, color="crimson", ls="--", lw=1.5, label="Newton $P_{16}(x)$")
    ax.plot(eval_pts, P_lagrange, color="royalblue", ls=":", lw=1.8, label="Lagrange $P_{16}(x)$")
    ax.scatter(x, y, color="green", s=15, zorder=5, label="data points (17)")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title("Q4: Newton vs Lagrange interpolation of 1/(1+x^2)\n(Runge phenomenon, degree 16)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("q4_function_plot.png", dpi=150)
    plt.close(fig)

    # Plot 2: E_N(x) & E_L(x)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.semilogy(eval_pts, E_N + 1e-18, color="crimson", label="$E_N(x)$ Newton")
    ax2.semilogy(eval_pts, E_L + 1e-18, color="royalblue", ls="--", label="$E_L(x)$ Lagrange")
    ax2.set_xlabel("x"); ax2.set_ylabel("absolute error (log)")
    ax2.set_title("Q4: Newton vs Lagrange error")
    ax2.legend(); ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig("q4_error_plot.png", dpi=150)
    plt.close(fig2)

    return E_N, E_L


if __name__ == "__main__":
    run()
