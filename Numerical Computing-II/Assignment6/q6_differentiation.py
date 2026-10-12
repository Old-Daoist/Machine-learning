"""
Q6. Numerical Differentiation Using Newton Interpolation
f(x) = sin x, using the Q2 data points (0, 0.2, 0.55, 1.0, 1.4).
f'(x) = P_n'(x) obtained by differentiating the Newton polynomial
        ALGORITHMICALLY (product rule loop -- see newton_derivative_eval).
Compared against:
    1. exact derivative cos(x)
    2. Lagrange-polynomial derivative (central diff on the Lagrange evaluator)
    3. plain central finite-difference derivative of f itself
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from newton_interp import (newton_coefficients, newton_derivative_eval,
                            lagrange_derivative_eval, central_difference_derivative)


def f(x):
    return math.sin(x)


def run():
    x = [0, 0.2, 0.55, 1.0, 1.4]
    y = [f(xi) for xi in x]
    coeffs = newton_coefficients(x, y)

    query = [0.3, 0.7, 1.1]
    print("=" * 70)
    print("Q6 -- Numerical differentiation comparison")
    print("=" * 70)
    header = f"{'x':>6}{'cos(x) exact':>15}{'Newton Pn´':>14}{'E_Newton':>12}" \
              f"{'Lagrange Pn´':>15}{'E_Lagr':>12}{'FiniteDiff':>13}{'E_FD':>12}"
    print(header)
    for xq in query:
        exact = math.cos(xq)
        pn = newton_derivative_eval(x, coeffs, xq)
        pl = lagrange_derivative_eval(x, y, xq)
        fd = central_difference_derivative(f, xq, h=1e-4)
        e_n = abs(exact - pn)
        e_l = abs(exact - pl)
        e_fd = abs(exact - fd)
        print(f"{xq:6.2f}{exact:15.8f}{pn:14.8f}{e_n:12.3e}"
              f"{pl:15.8f}{e_l:12.3e}{fd:13.8f}{e_fd:12.3e}")

        rel_n = e_n / abs(exact) if exact != 0 else float("nan")
        print(f"        relative error (Newton) = {rel_n:.3e}")

    # dense comparison plot
    xs_fine = np.linspace(0, 1.4, 300)
    exact_curve = np.cos(xs_fine)
    newton_curve = np.array([newton_derivative_eval(x, coeffs, xq) for xq in xs_fine])
    err_curve = np.abs(exact_curve - newton_curve)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs_fine, exact_curve, color="black", lw=2, label="exact $f'(x)=\\cos x$")
    ax.plot(xs_fine, newton_curve, color="crimson", ls="--", lw=1.5,
            label="Newton derivative $P_4'(x)$")
    ax.set_xlabel("x"); ax.set_ylabel("f'(x)")
    ax.set_title("Q6: Exact vs Newton-polynomial derivative of sin(x)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("q6_derivative_plot.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.plot(xs_fine, err_curve, color="darkorange")
    ax2.set_xlabel("x"); ax2.set_ylabel("E(x) = |cos x - Pn'(x)|")
    ax2.set_title("Q6: Derivative error over [0, 1.4]")
    ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("q6_derivative_error.png", dpi=150)
    plt.close(fig2)

    print(f"\nMax derivative error over [0,1.4]: {err_curve.max():.4e} "
          f"at x={xs_fine[err_curve.argmax()]:.4f}")


if __name__ == "__main__":
    run()
