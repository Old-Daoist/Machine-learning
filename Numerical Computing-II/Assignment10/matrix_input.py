"""Functions for reading matrices from terminal input or a file."""


def read_matrix():
    """Read an n x n matrix from terminal input."""
    try:
        n = int(input("Enter matrix order n: "))
    except ValueError as exc:
        raise ValueError("Matrix order must be a positive integer.") from exc

    if n <= 0:
        raise ValueError("Matrix order must be a positive integer.")

    print(f"Enter {n} rows, each with {n} numbers separated by spaces.")
    matrix = []
    for i in range(n):
        try:
            row = list(map(
                float,
                input(f"Row {i + 1}: ").replace(",", " ").split()
            ))
        except ValueError as exc:
            raise ValueError(
                f"Row {i + 1} contains a non-numeric value."
            ) from exc

        if len(row) != n:
            raise ValueError(f"Row {i + 1} must contain exactly {n} values.")
        matrix.append(row)

    return matrix


def read_matrix_from_file():
    """Load a square matrix from a .txt or .csv file.

    Use one matrix row per line, with comma- or whitespace-separated values.
    Do not include a header or the matrix order. Blank lines and lines beginning
    with # are ignored.
    """
    import tkinter as tk
    from tkinter import filedialog
    from pathlib import Path

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
        raise ValueError(
            "No file selected. Please run the program again and choose a file."
        )

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
        raise ValueError("The selected file is empty.")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError(
            f"The selected file must contain a square matrix: "
            f"{n} rows, each with {n} values."
        )

    print(f"Loaded matrix from: {path}")
    print("Matrix:")
    # NumPy is imported locally to keep this module's dependency explicit.
    import numpy as np
    print(np.array2string(np.asarray(matrix), precision=6, suppress_small=True))

    return matrix
