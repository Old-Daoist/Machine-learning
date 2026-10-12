import math
import matplotlib.pyplot as plt


class TrapezoidalIntegration:

    def __init__(self, function, lower_limit, upper_limit):
        self.function = function
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def calculate(self, number_of_intervals):
        # Width of each subinterval
        interval_width = (
            self.upper_limit - self.lower_limit
        ) / number_of_intervals

        # First and last function values are taken with half weight
        total = (
            self.function(self.lower_limit)
            + self.function(self.upper_limit)
        ) / 2

        # Add the interior function values
        for interval_number in range(1, number_of_intervals):
            x_value = (
                self.lower_limit
                + interval_number * interval_width
            )

            total += self.function(x_value)

        return interval_width * total

    def calculate_absolute_error(self, approximation, reference_value):
        return abs(reference_value - approximation)

    def calculate_accuracy(self, approximation, reference_value):
        absolute_error = self.calculate_absolute_error(
            approximation,
            reference_value
        )

        accuracy = (
            1 - absolute_error / abs(reference_value)
        ) * 100

        return accuracy


# ============================================================
# PROBLEM 1
# Integral of x^2 from 0 to 1
# ============================================================

def function_x_squared(x):
    return x ** 2


problem_1 = TrapezoidalIntegration(
    function_x_squared,
    0,
    1
)

# Exact value obtained analytically
exact_value_problem_1 = 1 / 3

# Required number of subintervals
number_of_intervals_problem_1 = [1, 2, 4, 8, 16]

approximations_problem_1 = []
errors_problem_1 = []
accuracies_problem_1 = []

print("\n" + "=" * 70)
print("PROBLEM 1: Integral of x^2 from 0 to 1")
print("=" * 70)

print("\nRequired calculations:")
print("-" * 70)
print(
    "Number of intervals\tApproximation\t\t"
    "Absolute Error\t\tAccuracy"
)
print("-" * 70)

for number_of_intervals in number_of_intervals_problem_1:

    approximation = problem_1.calculate(number_of_intervals)

    absolute_error = problem_1.calculate_absolute_error(
        approximation,
        exact_value_problem_1
    )

    accuracy = problem_1.calculate_accuracy(
        approximation,
        exact_value_problem_1
    )

    approximations_problem_1.append(approximation)
    errors_problem_1.append(absolute_error)
    accuracies_problem_1.append(accuracy)

    print(
        f"{number_of_intervals}\t\t\t"
        f"{approximation:.8f}\t\t"
        f"{absolute_error:.8f}\t\t"
        f"{accuracy:.6f}%"
    )

print("\nExact value =", exact_value_problem_1)


# Allow additional number of subintervals
while True:

    user_input = input(
        "\nEnter another number of intervals for Problem 1 "
        "(0 to continue): "
    )

    number_of_intervals = int(user_input)

    if number_of_intervals == 0:
        break

    if number_of_intervals < 1:
        print("Number of intervals must be greater than 0.")
        continue

    approximation = problem_1.calculate(number_of_intervals)

    absolute_error = problem_1.calculate_absolute_error(
        approximation,
        exact_value_problem_1
    )

    accuracy = problem_1.calculate_accuracy(
        approximation,
        exact_value_problem_1
    )

    number_of_intervals_problem_1.append(number_of_intervals)
    approximations_problem_1.append(approximation)
    errors_problem_1.append(absolute_error)
    accuracies_problem_1.append(accuracy)

    print("\nFor", number_of_intervals, "intervals:")
    print("Approximation =", approximation)
    print("Absolute error =", absolute_error)
    print("Accuracy =", accuracy, "%")


# Problem 1 error graph
plt.figure(figsize=(8, 5))

plt.plot(
    number_of_intervals_problem_1,
    errors_problem_1,
    marker="o"
)

plt.xlabel("Number of intervals")
plt.ylabel("Absolute error")
plt.title("Problem 1: Absolute Error vs Number of Intervals")
plt.grid(True)
plt.show()


# Problem 1 accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    number_of_intervals_problem_1,
    accuracies_problem_1,
    marker="o"
)

plt.xlabel("Number of intervals")
plt.ylabel("Accuracy (%)")
plt.title("Problem 1: Accuracy vs Number of Intervals")
plt.grid(True)
plt.show()


# ============================================================
# PROBLEM 2
# Integral of e^(-x^2) from 0 to 1
# ============================================================

def function_exp_minus_x_squared(x):
    return math.exp(-(x ** 2))


problem_2 = TrapezoidalIntegration(
    function_exp_minus_x_squared,
    0,
    1
)

# A very fine numerical calculation is used as the reference value
reference_value_problem_2 = problem_2.calculate(100000)

# Required number of subintervals
number_of_intervals_problem_2 = [4, 8, 16, 32]

approximations_problem_2 = []
errors_problem_2 = []
accuracies_problem_2 = []

print("\n" + "=" * 70)
print("PROBLEM 2: Integral of exp(-x^2) from 0 to 1")
print("=" * 70)

print("\nRequired calculations:")
print("-" * 70)
print(
    "Number of intervals\tApproximation\t\t"
    "Absolute Error\t\tAccuracy"
)
print("-" * 70)

for number_of_intervals in number_of_intervals_problem_2:

    approximation = problem_2.calculate(number_of_intervals)

    absolute_error = problem_2.calculate_absolute_error(
        approximation,
        reference_value_problem_2
    )

    accuracy = problem_2.calculate_accuracy(
        approximation,
        reference_value_problem_2
    )

    approximations_problem_2.append(approximation)
    errors_problem_2.append(absolute_error)
    accuracies_problem_2.append(accuracy)

    print(
        f"{number_of_intervals}\t\t\t"
        f"{approximation:.8f}\t\t"
        f"{absolute_error:.8f}\t\t"
        f"{accuracy:.6f}%"
    )

print("\nReference value =", reference_value_problem_2)


# Allow additional number of subintervals
while True:

    user_input = input(
        "\nEnter another number of intervals for Problem 2 "
        "(0 to continue): "
    )

    number_of_intervals = int(user_input)

    if number_of_intervals == 0:
        break

    if number_of_intervals < 1:
        print("Number of intervals must be greater than 0.")
        continue

    approximation = problem_2.calculate(number_of_intervals)

    absolute_error = problem_2.calculate_absolute_error(
        approximation,
        reference_value_problem_2
    )

    accuracy = problem_2.calculate_accuracy(
        approximation,
        reference_value_problem_2
    )

    number_of_intervals_problem_2.append(number_of_intervals)
    approximations_problem_2.append(approximation)
    errors_problem_2.append(absolute_error)
    accuracies_problem_2.append(accuracy)

    print("\nFor", number_of_intervals, "intervals:")
    print("Approximation =", approximation)
    print("Absolute error =", absolute_error)
    print("Accuracy =", accuracy, "%")


# Problem 2 error graph
plt.figure(figsize=(8, 5))

plt.plot(
    number_of_intervals_problem_2,
    errors_problem_2,
    marker="o"
)

plt.xlabel("Number of intervals")
plt.ylabel("Absolute error")
plt.title("Problem 2: Absolute Error vs Number of Intervals")
plt.grid(True)
plt.show()


# Problem 2 accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    number_of_intervals_problem_2,
    accuracies_problem_2,
    marker="o"
)

plt.xlabel("Number of intervals")
plt.ylabel("Accuracy (%)")
plt.title("Problem 2: Accuracy vs Number of Intervals")
plt.grid(True)
plt.show()


# ============================================================
# PROBLEM 3
# Experimental data
# ============================================================

experimental_x_values = [
    0,
    0.5,
    1.0,
    1.5,
    2.0
]

experimental_function_values = [
    1.00,
    1.65,
    2.70,
    4.48,
    7.39
]

problem_3 = TrapezoidalIntegration(
    None,
    experimental_x_values[0],
    experimental_x_values[-1]
)

experimental_interval_width = (
    experimental_x_values[1]
    - experimental_x_values[0]
)

experimental_integral = (
    experimental_interval_width
    * (
        experimental_function_values[0] / 2
        + sum(experimental_function_values[1:-1])
        + experimental_function_values[-1] / 2
    )
)

print("\n" + "=" * 70)
print("PROBLEM 3: Experimental Data")
print("=" * 70)

print("\nExperimental data:")
print("-" * 30)

for x_value, function_value in zip(
    experimental_x_values,
    experimental_function_values
):
    print(
        f"x = {x_value:<4}   f(x) = {function_value}"
    )

print(
    "\nWidth between data points =",
    experimental_interval_width
)

print(
    "Estimated integral =",
    experimental_integral
)


# Problem 3 graph
plt.figure(figsize=(8, 5))

plt.plot(
    experimental_x_values,
    experimental_function_values,
    marker="o"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Problem 3: Experimental Data")
plt.grid(True)
plt.show()


print("\n" + "=" * 70)
print("ALL THREE PROBLEMS COMPLETED")
print("=" * 70)