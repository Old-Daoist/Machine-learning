import numpy as np


class PowerMethod:
    def __init__(self, matrix, tolerance=1e-10,
                 max_iterations=1000):

        self.A = np.array(matrix, dtype=float)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix must be square.")

        if self.A.shape[0] == 0:
            raise ValueError("Matrix must not be empty.")

        self.eigenvalue = None
        self.eigenvector = None
        self.iterations = 0
        self.converged = False

    def solve(self):
        n = self.A.shape[0]

        # Initial vector
        x = np.ones(n)
        x = x / np.linalg.norm(x)

        previous = None

        for k in range(1, self.max_iterations + 1):

            # Multiply matrix by vector
            y = self.A @ x

            # Normalize the new vector
            norm_y = np.linalg.norm(y)

            if norm_y == 0:
                raise ValueError("Cannot continue iteration.")

            x = y / norm_y

            # Calculate eigenvalue estimate
            Ax = self.A @ x
            eigenvalue = np.dot(x, Ax) / np.dot(x, x)

            self.eigenvalue = eigenvalue
            self.eigenvector = x.copy()
            self.iterations = k

            # Check convergence
            if previous is not None:
                if abs(eigenvalue - previous) < self.tolerance:
                    self.converged = True
                    break

            previous = eigenvalue

        return self.eigenvalue, self.eigenvector

    def print_results(self):
        print("Dominant Eigenvalue:", self.eigenvalue)
        print("Eigenvector:", self.eigenvector)
        print("Iterations:", self.iterations)
        print("Converged:", self.converged)


# Main program
A = [
    [4, 1, 0],
    [1, 20, 1],
    [0, 1, 4]
]

pm = PowerMethod(A)
pm.solve()
pm.print_results()