import numpy as np


class SimpsonOneThird:

    def __init__(self, function, a, b, n):
        self.function = function
        self.a = a
        self.b = b
        self.n = n
        self.h = (b - a) / n

        self.x = np.array([a + i * self.h for i in range(n + 1)])
        self.y = np.array([function(x) for x in self.x])

    def display_table(self):
        print("\nSimpson's 1/3 Function Table")
        print("-" * 40)
        print(f"{'i':<5}{'x':<15}{'f(x)':<15}")
        print("-" * 40)

        for i in range(self.n + 1):
            print(f"{i:<5}{self.x[i]:<15.6f}{self.y[i]:<15.6f}")

    def calculate(self):

        if self.n % 2 != 0:
            print("\nSimpson's 1/3 Rule cannot be applied.")
            print("Reason: n must be even.")
            return None

        odd_sum = np.sum(self.y[1:-1:2])
        even_sum = np.sum(self.y[2:-1:2])

        result = (self.h / 3) * (
            self.y[0]
            + self.y[-1]
            + 4 * odd_sum
            + 2 * even_sum
        )

        return result