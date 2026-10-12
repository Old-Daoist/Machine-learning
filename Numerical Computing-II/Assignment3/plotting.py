"""Plot generation for Assignment 3."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _positive(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def generate_plots(test_functions, results, outdir):
    outdir = Path(outdir)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})

    # 1. Function plots
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = np.linspace(0, 2, 500)
    for item in test_functions:
        ax.plot(xs, [item["function"](x) for x in xs], label=item["name"])
    ax.axvline(1.0, linestyle="--", linewidth=1, label="Evaluation point x=1")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Test Functions for Numerical Differentiation")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "function_plots.png", dpi=220)
    plt.close(fig)

    # 2. Central difference
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, g in results.groupby("Function", sort=False):
        gg = g.sort_values("h", ascending=False)
        ax.loglog(gg["h"], _positive(gg["Absolute Error D(h)"]), marker="o", markersize=3.5, label=name)
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Central Difference Error Convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "central_difference_error.png", dpi=220)
    plt.close(fig)

    # 3. Richardson
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, g in results.groupby("Function", sort=False):
        gg = g.sort_values("h", ascending=False)
        ax.loglog(gg["h"], _positive(gg["Absolute Error R(h)"]), marker="o", markersize=3.5, label=name)
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Richardson Extrapolation Error Convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "richardson_error.png", dpi=220)
    plt.close(fig)

    # 4. Direct comparison
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, (name, g) in zip(axes, results.groupby("Function", sort=False)):
        gg = g.sort_values("h", ascending=False)
        ax.loglog(gg["h"], _positive(gg["Absolute Error D(h)"]), marker="o", markersize=3, label="Central Difference")
        ax.loglog(gg["h"], _positive(gg["Absolute Error R(h)"]), marker="s", markersize=3, label="Richardson")
        ax.set_title(name)
        ax.set_xlabel("h")
        ax.set_ylabel("Absolute Error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Central Difference vs Richardson Extrapolation", fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "method_comparison.png", dpi=220)
    plt.close(fig)

    # 5. Observed convergence order
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, (name, g) in zip(axes, results.groupby("Function", sort=False)):
        gg = g.sort_values("h", ascending=False)
        ax.semilogx(gg["h"], gg["Observed Order D"], marker="o", markersize=3, label="Central Difference")
        ax.semilogx(gg["h"], gg["Observed Order R"], marker="s", markersize=3, label="Richardson")
        ax.axhline(2, linestyle="--", linewidth=1, label="Theoretical p=2")
        ax.axhline(4, linestyle=":", linewidth=1, label="Theoretical p=4")
        ax.set_title(name)
        ax.set_xlabel("h")
        ax.set_ylabel("Observed order p")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Observed Convergence Order", fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "convergence_order.png", dpi=220)
    plt.close(fig)

    # 6. Round-off behavior, focused on smallest h
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, (name, g) in zip(axes, results.groupby("Function", sort=False)):
        gg = g.sort_values("h", ascending=False)
        ax.loglog(gg["h"], _positive(gg["Absolute Error D(h)"]), marker="o", markersize=3, label="Central Difference")
        ax.loglog(gg["h"], _positive(gg["Absolute Error R(h)"]), marker="s", markersize=3, label="Richardson")
        ax.set_title(name)
        ax.set_xlabel("h")
        ax.set_ylabel("Absolute Error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Round-off Behavior as h Approaches Machine Precision", fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "roundoff_behavior.png", dpi=220)
    plt.close(fig)
