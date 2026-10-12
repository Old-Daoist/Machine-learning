
import math
import matplotlib.pyplot as plt

from forward_difference import ForwardDifference
from backward_difference import BackwardDifference
from central_difference import CentralDifference


# Evaluation point and step sizes
x = 1.0
step_sizes = [10.0 ** (-i) for i in range(1, 7)]


# Initialize differentiation methods
methods = {
    "Forward Difference": ForwardDifference(),
    "Backward Difference": BackwardDifference(),
    "Central Difference": CentralDifference()
}


# Test functions and their exact derivatives
functions = {
    "exp(x)": (
        math.exp,
        math.exp
    ),
    "sin(x)": (
        math.sin,
        math.cos
    ),
    "x^2 - 2x + 1": (
        lambda x: x**2 - 2*x + 1,
        lambda x: 2*x - 2
    )
}


# Plot styles for clear comparison
styles = {
    "Forward Difference": {
        "color": "blue",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.5,
        "markersize": 7,
        "zorder": 3
    },
    "Backward Difference": {
        "color": "orange",
        "linestyle": "--",
        "marker": "x",
        "linewidth": 1.8,
        "markersize": 7,
        "zorder": 2
    },
    "Central Difference": {
        "color": "green",
        "linestyle": "-.",
        "marker": "s",
        "linewidth": 1.8,
        "markersize": 5,
        "zorder": 4
    }
}


# Store all calculated results
all_results = {}


# Perform numerical differentiation
for function_name, (f, exact_derivative) in functions.items():

    exact = exact_derivative(x)
    all_results[function_name] = {}

    print("\n" + "=" * 100)
    print(f"Function: {function_name} | Evaluation point x = {x}")
    print(f"Exact derivative: {exact:.12e}")
    print("=" * 100)

    print(
        f"{'Method':<22}"
        f"{'h':<12}"
        f"{'Numerical Derivative':<25}"
        f"{'Absolute Error':<20}"
    )

    for method_name, method in methods.items():

        numerical_values = []
        errors = []

        for h in step_sizes:

            numerical = method.derivative(f, x, h)
            error = abs(exact - numerical)

            numerical_values.append(numerical)
            errors.append(error)

            print(
                f"{method_name:<22}"
                f"{h:<12.0e}"
                f"{numerical:<25.12e}"
                f"{error:<20.12e}"
            )

        all_results[function_name][method_name] = {
            "h": step_sizes,
            "derivatives": numerical_values,
            "errors": errors
        }


# Generate log-log plots
for function_name, method_results in all_results.items():

    plt.figure(figsize=(10, 7))

    for method_name, results in method_results.items():

        hs = results["h"]
        errors = results["errors"]

        # Logarithmic axes cannot display zero or negative values.
        positive_points = [
            (h, error)
            for h, error in zip(hs, errors)
            if error > 0
        ]

        if not positive_points:
            continue

        plot_h, plot_errors = zip(*positive_points)

        plt.loglog(
            plot_h,
            plot_errors,
            label=method_name,
            **styles[method_name]
        )

    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute Error")
    plt.title(f"Numerical Differentiation Error: {function_name}")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()

    # Generate a filename for each plot
    if function_name == "exp(x)":
        filename = "expx_error_plot.png"

    elif function_name == "sin(x)":
        filename = "sinx_error_plot.png"

    else:
        filename = "x2_minus_2x_plus_1_error_plot.png"

    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved as {filename}")

    plt.show()
    plt.close()

print("\nAll calculations and plots completed successfully.")
