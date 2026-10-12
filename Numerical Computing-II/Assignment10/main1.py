
import numpy as np

class PowerMethod:

    def __init__(self, A):
        self.A = np.array(A, dtype=float)

    def solve(self):
        n = len(self.A)

        # Initial vector
        x = np.ones(n)

        for i in range(n):
            x[i] = x[i] / np.linalg.norm(x)

        for i in range(100):

            # Multiply matrix by vector
            y = self.A @ x

            # Normalize the vector
            y = y / np.linalg.norm(y)

            # Calculate eigenvalue
            eigenvalue = np.dot(y, self.A @ y)

            # Check if the result is stable
            if np.linalg.norm(y - x) < 0.000001:
                x = y
                break

            x = y

        return eigenvalue, x

    def print_results(self):
        eigenvalue, eigenvector = self.solve()

        print("Dominant Eigenvalue:", eigenvalue)
        print("Eigenvector:")
        print(eigenvector)


# Main program
A = [
    [4, 1, 0],
    [1, 20, 1],
    [0, 1, 4]
]

p = PowerMethod(A)
p.print_results()
