import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, pi, erf

from simpson_one_third import SimpsonOneThird
from simpson_three_eighth import SimpsonThreeEighth


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

def print_result(method, result, exact=None):

    print(f"\n{method}")

    if result is None:
        print("Not applicable")
        return

    print(f"Result = {result:.10f}")

    if exact is not None:
        error = abs(exact - result)
        print(f"Error  = {error:.10f}")


# ---------------------------------------------------------
# Plot for Problems 1 and 2
# ---------------------------------------------------------

def plot_problem(function, a, b, n, title):

    x_smooth = np.linspace(a, b, 500)
    y_smooth = function(x_smooth)

    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.array([function(value) for value in x])

    plt.figure(figsize=(10, 6))

    # Actual function
    plt.plot(
        x_smooth,
        y_smooth,
        label="Function"
    )

    # Integration points
    plt.scatter(
        x,
        y,
        label="Integration Points"
    )

    # -----------------------------------------------------
    # Trapezoidal approximation
    # -----------------------------------------------------

    plt.plot(
        x,
        y,
        linestyle="--",
        label="Trapezoidal"
    )

    # -----------------------------------------------------
    # Simpson 1/3 approximation
    # -----------------------------------------------------

    if n % 2 == 0:

        for i in range(0, n, 2):

            x_local = x[i:i + 3]
            y_local = y[i:i + 3]

            coefficients = np.polyfit(
                x_local,
                y_local,
                2
            )

            x_curve = np.linspace(
                x_local[0],
                x_local[-1],
                50
            )

            y_curve = np.polyval(
                coefficients,
                x_curve
            )

            plt.plot(
                x_curve,
                y_curve,
                linewidth=2,
                label="Simpson 1/3" if i == 0 else ""
            )

    # -----------------------------------------------------
    # Simpson 3/8 approximation
    # -----------------------------------------------------

    if n % 3 == 0:

        for i in range(0, n, 3):

            x_local = x[i:i + 4]
            y_local = y[i:i + 4]

            coefficients = np.polyfit(
                x_local,
                y_local,
                3
            )

            x_curve = np.linspace(
                x_local[0],
                x_local[-1],
                50
            )

            y_curve = np.polyval(
                coefficients,
                x_curve
            )

            plt.plot(
                x_curve,
                y_curve,
                linewidth=2,
                label="Simpson 3/8" if i == 0 else ""
            )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    plt.show()


# ---------------------------------------------------------
# Problem 3 graph
# ---------------------------------------------------------

def plot_experimental_data(x, y, title):

    plt.figure(figsize=(10, 6))

    # Experimental data
    plt.scatter(
        x,
        y,
        label="Experimental Data"
    )

    # Connect data points
    plt.plot(
        x,
        y,
        linestyle="--",
        label="Trapezoidal"
    )

    # -----------------------------------------------------
    # Simpson 1/3 approximation
    # -----------------------------------------------------

    n = len(x) - 1

    if n % 2 == 0:

        for i in range(0, n, 2):

            x_local = x[i:i + 3]
            y_local = y[i:i + 3]

            coefficients = np.polyfit(
                x_local,
                y_local,
                2
            )

            x_curve = np.linspace(
                x_local[0],
                x_local[-1],
                50
            )

            y_curve = np.polyval(
                coefficients,
                x_curve
            )

            plt.plot(
                x_curve,
                y_curve,
                linewidth=2,
                label="Simpson 1/3" if i == 0 else ""
            )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    plt.show()


# =========================================================
# PROBLEM 1
# =========================================================

def problem1():

    print("\n")
    print("=" * 60)
    print("PROBLEM 1")
    print("=" * 60)

    print("\nIntegral:")
    print("∫ x² dx from 0 to 1")

    # -----------------------------------------------------
    # Ask user for n
    # -----------------------------------------------------

    while True:

        try:
            n = int(input("\nEnter number of subintervals n: "))

            if n <= 0:
                print("n must be greater than 0.")
                continue

            break

        except ValueError:
            print("Please enter an integer.")


    def function(x):
        return x ** 2


    a = 0
    b = 1

    exact = 1 / 3


    # -----------------------------------------------------
    # Simpson 1/3
    # -----------------------------------------------------

    simpson_13 = SimpsonOneThird(
        function,
        a,
        b,
        n
    )

    result_13 = simpson_13.calculate()


    # -----------------------------------------------------
    # Simpson 3/8
    # -----------------------------------------------------

    simpson_38 = SimpsonThreeEighth(
        function,
        a,
        b,
        n
    )

    result_38 = simpson_38.calculate()


    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nExact Value =", exact)

    print_result(
        "Simpson's 1/3 Rule",
        result_13,
        exact
    )

    print_result(
        "Simpson's 3/8 Rule",
        result_38,
        exact
    )


    # -----------------------------------------------------
    # Display table
    # -----------------------------------------------------

    if result_13 is not None:
        simpson_13.display_table()

    elif result_38 is not None:
        simpson_38.display_table()


    # -----------------------------------------------------
    # Graph
    # -----------------------------------------------------

    plot_problem(
        function,
        a,
        b,
        n,
        f"Problem 1: x², n = {n}"
    )


# =========================================================
# PROBLEM 2
# =========================================================

def problem2():

    print("\n")
    print("=" * 60)
    print("PROBLEM 2")
    print("=" * 60)

    print("\nIntegral:")
    print("∫ e^(-x²) dx from 0 to 1")

    # -----------------------------------------------------
    # Ask user for n
    # -----------------------------------------------------

    while True:

        try:
            n = int(input("\nEnter number of subintervals n: "))

            if n <= 0:
                print("n must be greater than 0.")
                continue

            break

        except ValueError:
            print("Please enter an integer.")


    def function(x):
        return np.exp(-x ** 2)


    a = 0
    b = 1

    exact = (sqrt(pi) / 2) * erf(1)


    # -----------------------------------------------------
    # Simpson 1/3
    # -----------------------------------------------------

    simpson_13 = SimpsonOneThird(
        function,
        a,
        b,
        n
    )

    result_13 = simpson_13.calculate()


    # -----------------------------------------------------
    # Simpson 3/8
    # -----------------------------------------------------

    simpson_38 = SimpsonThreeEighth(
        function,
        a,
        b,
        n
    )

    result_38 = simpson_38.calculate()


    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nExact Value =", exact)

    print_result(
        "Simpson's 1/3 Rule",
        result_13,
        exact
    )

    print_result(
        "Simpson's 3/8 Rule",
        result_38,
        exact
    )


    # -----------------------------------------------------
    # Display table
    # -----------------------------------------------------

    if result_13 is not None:
        simpson_13.display_table()

    elif result_38 is not None:
        simpson_38.display_table()


    # -----------------------------------------------------
    # Graph
    # -----------------------------------------------------

    plot_problem(
        function,
        a,
        b,
        n,
        f"Problem 2: e^(-x²), n = {n}"
    )


# =========================================================
# PROBLEM 3
# =========================================================

def problem3():

    print("\n")
    print("=" * 60)
    print("PROBLEM 3")
    print("=" * 60)

    print("\nExperimental Data")

    x_data = np.array([
        0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    y_data = np.array([
        1.00,
        1.65,
        2.70,
        4.48,
        7.39
    ])

    n = len(x_data) - 1

    a = x_data[0]
    b = x_data[-1]


    # -----------------------------------------------------
    # Function using interpolation
    # -----------------------------------------------------

    def data_function(x):
        return np.interp(
            x,
            x_data,
            y_data
        )


    # -----------------------------------------------------
    # Simpson 1/3
    # -----------------------------------------------------

    simpson_13 = SimpsonOneThird(
        data_function,
        a,
        b,
        n
    )

    result_13 = simpson_13.calculate()


    # -----------------------------------------------------
    # Simpson 3/8
    # -----------------------------------------------------

    simpson_38 = SimpsonThreeEighth(
        data_function,
        a,
        b,
        n
    )

    result_38 = simpson_38.calculate()


    print("\nNumber of subintervals:", n)
    print("Step size h:", (b - a) / n)


    print_result(
        "Simpson's 1/3 Rule",
        result_13
    )

    print_result(
        "Simpson's 3/8 Rule",
        result_38
    )


    print("\nNote:")
    print("The experimental data has n = 4.")
    print("Simpson's 1/3 can be applied because 4 is even.")
    print("Simpson's 3/8 cannot be applied because 4 is not")
    print("a multiple of 3.")

    print("\nNo exact error is calculated because")
    print("the exact underlying function is not provided.")


    # -----------------------------------------------------
    # Display table
    # -----------------------------------------------------

    simpson_13.display_table()


    # -----------------------------------------------------
    # Graph
    # -----------------------------------------------------

    plot_experimental_data(
        x_data,
        y_data,
        "Problem 3: Experimental Data"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    problem1()

    problem2()

    problem3()