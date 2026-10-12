import numpy as np


class SimpsonThreeEighth:

    def __init__(self, function, a, b, n):
        self.function = function
        self.a = a
        self.b = b
        self.n = n
        self.h = (b - a) / n

        self.x = np.array([a + i * self.h for i in range(n + 1)])
        self.y = np.array([function(x) for x in self.x])

    def display_table(self):
        print("\nSimpson's 3/8 Function Table")
        print("-" * 40)
        print(f"{'i':<5}{'x':<15}{'f(x)':<15}")
        print("-" * 40)

        for i in range(self.n + 1):
            print(f"{i:<5}{self.x[i]:<15.6f}{self.y[i]:<15.6f}")

    def calculate(self):

        if self.n % 3 != 0:
            print("\nSimpson's 3/8 Rule cannot be applied.")
            print("Reason: n must be a multiple of 3.")
            return None

        total = self.y[0] + self.y[-1]

        for i in range(1, self.n):

            if i % 3 == 0:
                total += 2 * self.y[i]
            else:
                total += 3 * self.y[i]

        result = (3 * self.h / 8) * total

        return result