"""Power Method for estimating a dominant eigenvalue and eigenvector."""

import numpy as np


class PowerMethod:
    """Estimate a matrix's dominant eigenvalue and corresponding eigenvector."""

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
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be a finite number greater than zero.")
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
        self.eigenvalue = None
        self.eigenvector = None
        self.iterations = 0

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
                "residual": residual,
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
        """Display iteration history and the final result."""
        if self.eigenvalue is None:
            self.solve()

        print("\nIteration history")
        print(f"{'k':>4} {'Eigenvalue estimate':>22} {'Residual':>14}")
        for item in self.history:
            print(
                f"{item['iteration']:>4} "
                f"{item['eigenvalue']:>22.10f} "
                f"{item['residual']:>14.3e}"
            )

        print("\nEstimated dominant eigenvalue:", f"{self.eigenvalue:.10f}")
        print("Estimated eigenvector (unit length):")
        print(np.array2string(self.eigenvector, precision=8, suppress_small=True))
        print("Iterations:", self.iterations)
        print("Converged:", self.converged)
        print("Final residual:", f"{self.history[-1]['residual']:.3e}")
