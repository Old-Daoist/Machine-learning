"""
Power Method using OOP, with convergence graphs.

Requirements:
    python -m pip install numpy matplotlib

Run:
    python assignment_x_power_method_oop.py

The program estimates the dominant eigenvalue and corresponding eigenvector.
It does not generally calculate every eigenvalue.
"""

import numpy as np
import matplotlib.pyplot as plt


class PowerMethod:
    """Estimate a matrix's dominant eigenvalue and eigenvector."""

    def __init__(self, matrix, initial_vector=None, tolerance=1e-8,
                 max_iterations=1000):
        self.A = np.asarray(matrix, dtype=float)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix must be a square n x n matrix.")
        if self.A.shape[0] == 0:
            raise ValueError("Matrix cannot be empty.")
        if not np.all(np.isfinite(self.A)):
            raise ValueError("Matrix entries must be finite numbers.")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be greater than zero.")
        if self.max_iterations <= 0:
            raise ValueError("Maximum iterations must be positive.")

        self.n = self.A.shape[0]

        if initial_vector is None:
            x = np.ones(self.n, dtype=float)
        else:
            x = np.asarray(initial_vector, dtype=float)
            if x.ndim != 1 or len(x) != self.n:
                raise ValueError("Initial vector must contain exactly n values.")
            if not np.all(np.isfinite(x)):
                raise ValueError("Initial vector entries must be finite numbers.")

        x_norm = np.linalg.norm(x)
        if x_norm == 0:
            raise ValueError("Initial vector cannot be the zero vector.")

        self.initial_vector = x / x_norm
        self.eigenvalue = None
        self.eigenvector = None
        self.iterations = 0
        self.converged = False
        self.history = []

    def _estimate_eigenvalue(self, vector):
        """Estimate eigenvalue using the Rayleigh quotient."""
        Av = self.A @ vector
        return float(np.dot(vector, Av) / np.dot(vector, vector))

    def _calculate_residual(self, eigenvalue, vector):
        """Calculate ||A v - lambda v||_2."""
        return float(np.linalg.norm(self.A @ vector - eigenvalue * vector))

    def solve(self):
        """Run the Power Method and store iteration history."""
        x = self.initial_vector.copy()
        previous_eigenvalue = None
        self.history = []
        self.converged = False

        for k in range(1, self.max_iterations + 1):
            y = self.A @ x
            y_norm = np.linalg.norm(y)

            if y_norm == 0:
                raise ValueError(
                    "Matrix-vector product became zero; the method cannot continue."
                )

            x = y / y_norm
            eigenvalue = self._estimate_eigenvalue(x)
            residual = self._calculate_residual(eigenvalue, x)

            self.history.append({
                "iteration": k,
                "eigenvalue": eigenvalue,
                "eigenvector": x.copy(),
                "residual": residual
            })

            eigenvalue_stable = (
                previous_eigenvalue is not None
                and abs(eigenvalue - previous_eigenvalue) < self.tolerance
            )

            self.eigenvalue = eigenvalue
            self.eigenvector = x.copy()
            self.iterations = k

            if eigenvalue_stable and residual < self.tolerance:
                self.converged = True
                break

            previous_eigenvalue = eigenvalue

        return self.eigenvalue, self.eigenvector

    def display_results(self):
        """Display iteration table and final result."""
        if self.eigenvalue is None:
            self.solve()

        print("\\nIteration history")
        print(f"{'k':>4} {'Eigenvalue estimate':>22} {'Residual':>14}")
        for item in self.history:
            print(f"{item['iteration']:>4} "
                  f"{item['eigenvalue']:>22.10f} "
                  f"{item['residual']:>14.3e}")

        print("\\nEstimated dominant eigenvalue:", f"{self.eigenvalue:.10f}")
        print("Estimated eigenvector (unit length):")
        print(np.array2string(self.eigenvector, precision=8, suppress_small=True))
        print("Iterations:", self.iterations)
        print("Converged:", self.converged)
        print("Final residual:", f"{self.history[-1]['residual']:.3e}")

    def plot_convergence(self):
        """Plot eigenvalue estimate and residual versus iteration."""
        if not self.history:
            self.solve()

        iterations = [item["iteration"] for item in self.history]
        estimates = [item["eigenvalue"] for item in self.history]
        residuals = [item["residual"] for item in self.history]

        # Graph 1: eigenvalue estimate as iterations increase.
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, estimates, marker="o", label="Eigenvalue estimate")
        plt.axhline(self.eigenvalue, linestyle="--",
                    label=f"Final estimate = {self.eigenvalue:.8f}")
        plt.xlabel("Iteration")
        plt.ylabel("Eigenvalue estimate")
        plt.title("Power Method: Eigenvalue Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Graph 2: residual on a logarithmic y-axis.
        plt.figure(figsize=(8, 5))
        safe_residuals = [max(value, np.finfo(float).tiny) for value in residuals]
        plt.semilogy(iterations, safe_residuals, marker="o", label="Residual")
        plt.axhline(self.tolerance, linestyle="--", label="Tolerance")
        plt.xlabel("Iteration")
        plt.ylabel("Residual ||Ax - λx||₂ (log scale)")
        plt.title("Power Method: Residual Convergence")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.show()


def read_matrix():
    """Read an n x n matrix from terminal input."""
    n = int(input("Enter matrix order n: "))
    if n <= 0:
        raise ValueError("Matrix order must be a positive integer.")

    print(f"Enter {n} rows, each with {n} numbers separated by spaces.")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Row {i + 1}: ").replace(",", " ").split()))
        if len(row) != n:
            raise ValueError(f"Row {i + 1} must contain exactly {n} values.")
        matrix.append(row)
    return matrix


def read_matrix_from_file():
    """Open a file-picker window and load a matrix from a .txt or .csv file.

    The file should contain one matrix row per line. Values may be separated
    by commas or whitespace. Do not include a header or the matrix order.
    Example file contents:
        4, 1, 0
        1, 20, 1
        0, 1, 4
    """
    import tkinter as tk
    from tkinter import filedialog
    from pathlib import Path

    # Hide the small empty Tk window; show the file browser instead.
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            parent=root,
            title="Select a matrix file",
            filetypes=[
                ("Matrix files", "*.txt *.csv"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not filename:
        raise ValueError("No file selected. Please run the program again and choose a file.")

    path = Path(filename)
    matrix = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.replace(",", " ").split()
            try:
                row = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} in {path.name} contains a non-numeric value."
                ) from exc
            matrix.append(row)

    if not matrix:
        raise ValueError("The selected matrix file is empty.")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError(
            f"The selected file must contain a square matrix: {n} rows, each with {n} values."
        )

    print(f"Loaded matrix from: {path}")
    print("Matrix:")
    print(np.array2string(np.asarray(matrix), precision=6, suppress_small=True))
    return matrix


def main():
    print("POWER METHOD (OOP + CONVERGENCE GRAPHS)")
    print("1. Enter a square matrix manually")
    print("2. Browse for a matrix file (.txt or .csv)")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        matrix = read_matrix()
    elif choice == "2":
        matrix = read_matrix_from_file()
    else:
        print("Invalid choice. Please choose 1 or 2.")
        return

    tolerance_text = input("Tolerance (Enter for 1e-8): ").strip()
    tolerance = float(tolerance_text) if tolerance_text else 1e-8
    max_text = input("Maximum iterations (Enter for 1000): ").strip()
    max_iterations = int(max_text) if max_text else 1000

    method = PowerMethod(matrix, tolerance=tolerance,
                         max_iterations=max_iterations)
    method.solve()
    method.display_results()
    method.plot_convergence()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError) as error:
        print("Input error:", error)
