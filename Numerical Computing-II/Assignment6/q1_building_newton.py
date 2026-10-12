"""
Q1. Building Newton's Polynomial
f(x) = x^2 + 2x + 1, data at x = -2,-1,0,1,2
"""
from newton_interp import (divided_difference_table, newton_coefficients,
                            newton_eval, print_dd_table)


def f(x):
    return x**2 + 2*x + 1


def run():
    x = [-2, -1, 0, 1, 2]
    y = [f(xi) for xi in x]

    print("=" * 60)
    print("Q1 -- Divided-difference table")
    print("=" * 60)
    print_dd_table(x, y)

    coeffs = newton_coefficients(x, y)
    print("\nNewton coefficients [a0..a4]:")
    for i, c in enumerate(coeffs):
        print(f"  a{i} = {c:.10g}")

    print("\nP4(x) evaluated at the query points:")
    query = [-1.5, -0.5, 0.5, 1.5]
    print(f"{'x':>8} {'f(x)':>12} {'P4(x)':>14} {'|f(x)-P4(x)|':>16}")
    for xq in query:
        pv = newton_eval(x, coeffs, xq)
        fv = f(xq)
        print(f"{xq:8.3f} {fv:12.6f} {pv:14.10f} {abs(fv - pv):16.3e}")

    return x, y, coeffs


if __name__ == "__main__":
    run()
