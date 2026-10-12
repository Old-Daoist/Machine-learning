"""Main entry point for the Power Method assignment."""

from matrix_input import read_matrix, read_matrix_from_file
from power_method import PowerMethod
from plotting import plot_convergence


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
    try:
        tolerance = float(tolerance_text) if tolerance_text else 1e-8
    except ValueError as exc:
        raise ValueError("Tolerance must be a number.") from exc

    max_text = input("Maximum iterations (Enter for 1000): ").strip()
    try:
        max_iterations = int(max_text) if max_text else 1000
    except ValueError as exc:
        raise ValueError("Maximum iterations must be an integer.") from exc

    method = PowerMethod(
        matrix,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    method.solve()
    method.display_results()
    plot_convergence(method)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, OverflowError) as error:
        print("Input error:", error)
