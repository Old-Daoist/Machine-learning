
import math
import matplotlib.pyplot as plt

from central_difference import CentralDifference
from richardson_extrapolation import RichardsonExtrapolation

# Evaluation point and step sizes
x = 1.0
step_sizes = [10.0 ** (-i) for i in range(1, 11)]

central = CentralDifference()
richardson = RichardsonExtrapolation()

methods = {
    "Central Difference": central,
    "Richardson Extrapolation": richardson
}

functions = {
    "exp(x)": (math.exp, math.exp),
    "sin(x)": (math.sin, math.cos),
    "cos(x)": (math.cos, lambda x: -math.sin(x)),
    "x^2 - 2x + 1": (
        lambda x: x**2 - 2*x + 1,
        lambda x: 2*x - 2
    )
}

styles = {
    "Central Difference": {
        "color": "blue",
        "linestyle": "-",
        "marker": "o"
    },
    "Richardson Extrapolation": {
        "color": "red",
        "linestyle": "--",
        "marker": "s"
    }
}

all_results = {}

# Calculate and print results
for name, (f, exact_function) in functions.items():
    exact = exact_function(x)
    all_results[name] = {}

    print("\n" + "=" * 100)
    print(f"Function: {name} | x = {x}")
    print(f"Exact derivative: {exact:.12e}")
    print("=" * 100)

    print(
        f"{'Method':<27}{'h':<12}"
        f"{'Approximation':<25}{'Absolute Error':<22}"
    )

    for method_name, method in methods.items():
        approximations = []
        errors = []

        for h in step_sizes:
            approximation = method.derivative(f, x, h)
            error = abs(exact - approximation)

            approximations.append(approximation)
            errors.append(error)

            print(
                f"{method_name:<27}{h:<12.0e}"
                f"{approximation:<25.12e}{error:<22.12e}"
            )

        all_results[name][method_name] = {
            "h": step_sizes,
            "approximations": approximations,
            "errors": errors
        }

# Generate and save all four plots
for name, method_results in all_results.items():
    fig, ax = plt.subplots(figsize=(10, 7))

    for method_name, results in method_results.items():
        points = sorted(
            [
                (h, error)
                for h, error in zip(
                    results["h"], results["errors"]
                )
                if error > 0
            ],
            key=lambda point: point[0]
        )

        if not points:
            print(f"No positive errors to plot for {name}, {method_name}")
            continue

        hs, errors = zip(*points)

        ax.loglog(
            hs,
            errors,
            label=method_name,
            linewidth=2,
            markersize=5,
            **styles[method_name]
        )

    ax.set_xlabel("Step size (h)")
    ax.set_ylabel("Absolute Error")
    ax.set_title(
        f"Richardson Extrapolation vs Central Difference: {name}"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if name == "exp(x)":
        filename = "expx_richardson_plot.png"
    elif name == "sin(x)":
        filename = "sinx_richardson_plot.png"
    elif name == "cos(x)":
        filename = "cosx_richardson_plot.png"
    else:
        filename = "quadratic_richardson_plot.png"

    fig.savefig(filename, dpi=300)
    print(f"\nPlot saved: {filename}")
    plt.show()
    plt.close(fig)

# Calculate adjacent observed slopes
print("\n" + "=" * 80)
print("OBSERVED CONVERGENCE SLOPES")
print("=" * 80)

for name, method_results in all_results.items():
    print(f"\nFunction: {name}")

    for method_name, results in method_results.items():
        hs = results["h"]
        errors = results["errors"]

        print(f"\n{method_name}:")

        # Sort pairs so h increases from smallest to largest
        pairs = sorted(zip(hs, errors), key=lambda pair: pair[0])

        for (h1, e1), (h2, e2) in zip(pairs, pairs[1:]):
            if e1 > 0 and e2 > 0:
                slope = math.log(e2 / e1) / math.log(h2 / h1)
                print(
                    f"h: {h1:.0e} -> {h2:.0e}, "
                    f"slope = {slope:.4f}"
                )
            else:
                print(
                    f"h: {h1:.0e} -> {h2:.0e}, "
                    "slope unavailable because an error is zero"
                )

print("\nAll calculations and plots completed.")
