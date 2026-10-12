"""
newton_interp.py
=================
General-purpose Newton divided-difference interpolation toolkit.

Implements, from scratch (no numpy.polyfit / scipy.interpolate / etc.):
    - divided_difference_table(x, y)   -> full lower-triangular DD table
    - newton_coefficients(x, y)        -> the Newton coefficients a0..an
    - newton_eval(x_data, coeffs, xq)  -> evaluate P_n(xq) (nested/Horner form)
    - newton_derivative_eval(...)      -> evaluate P_n'(xq) algorithmically
                                           (product rule applied term-by-term,
                                           NOT symbolic differentiation)
    - lagrange_eval(x, y, xq)          -> Lagrange form, for comparison in Q4
    - print_dd_table(...)              -> pretty-print the divided-difference
                                           table used in the written report

Works for ANY number of data points (not hard-coded to 5, 6, 17, ... points).

Author: Khumba Lunganlung -- Numerical Computing II, Assignment VI
"""

from __future__ import annotations
from typing import Sequence, List


# ---------------------------------------------------------------------------
# 1. Divided-difference table
# ---------------------------------------------------------------------------
def divided_difference_table(x: Sequence[float], y: Sequence[float]) -> List[List[float]]:
    """
    Build the full divided-difference table.

    Returns a list of lists `F` where F[i][j] is the j-th divided difference
    that uses points x[i], x[i+1], ..., x[i+j]   (0-based).
    F[0][0..n] (i.e. F[0][j] for j = 0..n) are exactly the Newton
    coefficients a0, a1, ..., an, since the Newton polynomial is built by
    always advancing from the FIRST point.

    n = len(x) - 1 is the polynomial degree.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    if n < 1:
        raise ValueError("need at least one data point")

    # F[i][j], i = 0..n-1 (row), j = 0..n-1-i (order of the difference)
    F = [[0.0] * n for _ in range(n)]
    for i in range(n):
        F[i][0] = float(y[i])

    for j in range(1, n):          # order of divided difference
        for i in range(n - j):     # starting index
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])

    return F


def newton_coefficients(x: Sequence[float], y: Sequence[float]) -> List[float]:
    """Return [a0, a1, ..., an] = [F[0][0], F[0][1], ..., F[0][n]]."""
    F = divided_difference_table(x, y)
    n = len(x)
    return [F[0][j] for j in range(n)]


def print_dd_table(x: Sequence[float], y: Sequence[float]) -> None:
    """Pretty-print the divided-difference table (lower-triangular form)."""
    F = divided_difference_table(x, y)
    n = len(x)
    header = "   x     " + "".join(f"f[{j}]".rjust(12) for j in range(n))
    print(header)
    for i in range(n):
        row = f"{x[i]:8.4f} "
        for j in range(n - i):
            row += f"{F[i][j]:12.6f}"
        print(row)


# ---------------------------------------------------------------------------
# 2. Evaluating the Newton polynomial (nested multiplication, like Horner's
#    rule but for the Newton basis) -- O(n) per evaluation point once the
#    coefficients are known.
# ---------------------------------------------------------------------------
def newton_eval(x_data: Sequence[float], coeffs: Sequence[float], xq: float) -> float:
    """
    Evaluate P_n(xq) = a0 + a1(xq-x0) + a2(xq-x0)(xq-x1) + ...
    using nested (Horner-like) evaluation:
        P = a_n
        P = a_{n-1} + (xq - x_{n-1}) * P
        ...
        P = a_0 + (xq - x_0) * P
    """
    n = len(coeffs)
    result = coeffs[n - 1]
    for k in range(n - 2, -1, -1):
        result = coeffs[k] + (xq - x_data[k]) * result
    return result


def newton_eval_array(x_data: Sequence[float], coeffs: Sequence[float], xs) -> List[float]:
    """Vectorised convenience wrapper (loops internally -- still no built-in
    interpolation routine is used)."""
    return [newton_eval(x_data, coeffs, xq) for xq in xs]


# ---------------------------------------------------------------------------
# 3. Differentiating the Newton polynomial ALGORITHMICALLY.
#
#    P_n(x) = sum_{i=0}^{n} a_i * prod_{k=0}^{i-1} (x - x_k)
#
#    Let phi_i(x) = prod_{k=0}^{i-1} (x - x_k)   (phi_0 = 1)
#    By the product rule:
#        phi_i'(x) = phi_i(x) * sum_{k=0}^{i-1} 1/(x - x_k)     [x != x_k]
#    which is numerically fine here since we only ever evaluate the
#    derivative at points that are NOT among the data nodes.
#    We compute this term-by-term (an explicit loop, not sympy/symbolic diff).
# ---------------------------------------------------------------------------
def newton_derivative_eval(x_data: Sequence[float], coeffs: Sequence[float], xq: float) -> float:
    """
    Evaluate P_n'(xq) by differentiating the Newton form term by term using
    the product rule, computed with an explicit algorithm (no symbolic
    differentiation library).
    """
    n = len(coeffs)  # number of coefficients = degree + 1
    deriv = 0.0
    phi_prev = 1.0   # phi_0(x) = 1
    dphi_prev = 0.0  # derivative of phi_0 is 0

    for i in range(n):
        if i == 0:
            # derivative of the constant term a0 * phi_0 = a0 * 1 is 0
            phi_i = 1.0
            dphi_i = 0.0
        else:
            # phi_i = phi_{i-1} * (xq - x_data[i-1])
            phi_i = phi_prev * (xq - x_data[i - 1])
            # d/dx [phi_{i-1}*(x-x_{i-1})] = dphi_{i-1}*(x-x_{i-1}) + phi_{i-1}
            dphi_i = dphi_prev * (xq - x_data[i - 1]) + phi_prev

        deriv += coeffs[i] * dphi_i
        phi_prev = phi_i
        dphi_prev = dphi_i

    return deriv


def newton_derivative_eval_array(x_data, coeffs, xs) -> List[float]:
    return [newton_derivative_eval(x_data, coeffs, xq) for xq in xs]


# ---------------------------------------------------------------------------
# 4. Lagrange interpolation (independent implementation, used only for the
#    Newton-vs-Lagrange comparison in Q4). Also O(n^2) per point, built from
#    scratch.
# ---------------------------------------------------------------------------
def lagrange_eval(x: Sequence[float], y: Sequence[float], xq: float) -> float:
    n = len(x)
    total = 0.0
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (xq - x[j]) / (x[i] - x[j])
        total += y[i] * Li
    return total


def lagrange_eval_array(x, y, xs) -> List[float]:
    return [lagrange_eval(x, y, xq) for xq in xs]


# ---------------------------------------------------------------------------
# 5. Simple central-difference numerical derivative (used only as an extra
#    comparison method in Q6 -- "Finite Difference vs Lagrange vs Newton").
# ---------------------------------------------------------------------------
def central_difference_derivative(f, xq: float, h: float = 1e-4) -> float:
    return (f(xq + h) - f(xq - h)) / (2 * h)


def lagrange_derivative_eval(x: Sequence[float], y: Sequence[float], xq: float, h: float = 1e-6) -> float:
    """
    Differentiate the Lagrange polynomial. We do this algorithmically via a
    tiny central-difference on the Lagrange evaluator itself (the Lagrange
    polynomial is the SAME polynomial as the Newton one, so this is only used
    as a cross-check in Q6, not as the primary "Newton derivative" method).
    """
    return (lagrange_eval(x, y, xq + h) - lagrange_eval(x, y, xq - h)) / (2 * h)


if __name__ == "__main__":
    # tiny self-test
    xs = [-2, -1, 0, 1, 2]
    ys = [1, 0, 1, 4, 9]
    c = newton_coefficients(xs, ys)
    print("coeffs:", c)
    print("P(0.5) =", newton_eval(xs, c, 0.5), " exact =", 0.5**2 + 2*0.5 + 1)
    print("P'(0.5) =", newton_derivative_eval(xs, c, 0.5), " exact = 3.0")
