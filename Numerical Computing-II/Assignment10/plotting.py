"""Convergence plots for the Power Method."""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(method):
    """Plot eigenvalue estimates and residuals for a solved PowerMethod object."""
    if not method.history:
        method.solve()

    iterations = [item["iteration"] for item in method.history]
    estimates = [item["eigenvalue"] for item in method.history]
    residuals = [item["residual"] for item in method.history]

    # Graph 1: eigenvalue estimate as iterations increase.
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, estimates, marker="o", label="Eigenvalue estimate")
    plt.axhline(
        method.eigenvalue,
        linestyle="--",
        label=f"Final estimate = {method.eigenvalue:.8f}",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalue estimate")
    plt.title("Power Method: Eigenvalue Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Graph 2: residual on a logarithmic y-axis.
    plt.figure(figsize=(8, 5))
    safe_residuals = [
        max(value, np.finfo(float).tiny) for value in residuals
    ]
    plt.semilogy(iterations, safe_residuals, marker="o", label="Residual")
    plt.axhline(method.tolerance, linestyle="--", label="Tolerance")
    plt.xlabel("Iteration")
    plt.ylabel("Residual ||Av - λv||₂ (log scale)")
    plt.title("Power Method: Residual Convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
