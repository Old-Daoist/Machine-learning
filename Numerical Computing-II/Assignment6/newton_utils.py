import numpy as np


def divided_difference_table(x, y):
    """
    Construct Newton's divided-difference table.

    Parameters
    ----------
    x : array-like
        Interpolation points.
    y : array-like
        Function values.

    Returns
    -------
    table : ndarray
        Divided-difference table.
        First column contains y values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)

    if len(y) != n:
        raise ValueError("x and y must have the same length.")

    if len(np.unique(x)) != n:
        raise ValueError("Interpolation points must be distinct.")

    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (
                table[i + 1, j - 1] - table[i, j - 1]
            ) / (x[i + j] - x[i])

    return table


def newton_coefficients(table):
    """
    Extract Newton coefficients from the divided-difference table.
    """
    return table[0, :]


def newton_interpolate(x_data, coefficients, x):
    """
    Evaluate Newton interpolation polynomial.

    Uses nested multiplication, so it is efficient and
    works for scalar or array x.
    """
    x_data = np.asarray(x_data, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)

    x = np.asarray(x, dtype=float)

    n = len(coefficients)

    result = np.full_like(x, coefficients[-1], dtype=float)

    for k in range(n - 2, -1, -1):
        result = coefficients[k] + (x - x_data[k]) * result

    return result


def newton_derivative(x_data, coefficients, x):
    """
    Evaluate the derivative of the Newton polynomial.

    The derivative is obtained algorithmically from the
    Newton polynomial, without symbolic differentiation.
    """
    x_data = np.asarray(x_data, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    x = np.asarray(x, dtype=float)

    n = len(coefficients)

    # Start with highest-order coefficient
    p = np.full_like(x, coefficients[-1], dtype=float)
    dp = np.zeros_like(x, dtype=float)

    for k in range(n - 2, -1, -1):
        dp = dp * (x - x_data[k]) + p
        p = p * (x - x_data[k]) + coefficients[k]

    return dp


def print_table(x, table):
    """
    Print divided-difference table neatly.
    """
    n = len(x)

    print("\nDivided-Difference Table")
    print("-" * 80)

    headers = ["x", "f[x]"]

    for j in range(1, n):
        headers.append(f"DD{j}")

    print(" ".join(f"{h:>15}" for h in headers))

    for i in range(n):
        row = [x[i]]

        for j in range(n - i):
            row.append(table[i, j])

        print(" ".join(f"{value:15.8f}" for value in row))

    print("-" * 80)


def print_coefficients(coefficients):
    print("\nNewton Coefficients:")
    for i, c in enumerate(coefficients):
        print(f"a{i} = {c:.12f}")