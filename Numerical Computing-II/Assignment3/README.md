# Numerical Computing II — Assignment 3
## Can we make Numerical Differentiation more accurate?
### Investigation of Richardson Extrapolation

This project compares the second-order **Central Difference** formula with fourth-order **Richardson Extrapolation** at `x = 1`.

## Objective

The assignment investigates whether Richardson extrapolation can remove the leading `O(h^2)` truncation-error term from the central-difference approximation and produce approximately `O(h^4)` behavior. It also investigates the eventual effect of floating-point round-off and subtractive cancellation.

## Numerical formulas

Central Difference:

`D(h) = [f(x+h) - f(x-h)] / (2h)`

Richardson Extrapolation:

`R(h) = [4D(h/2) - D(h)] / 3`

The Richardson approximation uses a fresh central difference at `h/2` for every tested `h`.

## Test functions

At `x = 1`:

| Function | Exact derivative |
|---|---|
| `f(x) = e^x` | `f'(x) = e^x` |
| `f(x) = sin(x)` | `f'(x) = cos(x)` |
| `f(x) = cos(x)` | `f'(x) = -sin(x)` |
| `f(x) = x^3 - 2x + 1` | `f'(x) = 3x^2 - 2` |

The derivative functions are explicitly defined in `functions.py`; exact values are evaluated at runtime rather than hard-coded.

## Step sizes

The calculation uses double-precision Python floating-point arithmetic and tests:

`h = 10^-1, 10^-2, ..., 10^-16`

Very small steps are intentionally included so the transition from truncation error to floating-point round-off can be observed.

## Errors

For each function and each `h`:

`Absolute Error = |exact derivative - numerical approximation|`

Both central-difference and Richardson errors are calculated.

## Observed convergence order

For consecutive step sizes:

`p = log(E_i / E_{i+1}) / log(h_i / h_{i+1})`

Because `h` is reduced by a factor of 10:

`p = log10(E_i / E_{i+1})`

The expected truncation-error orders are:

- Central Difference: `O(h^2)`, theoretical slope `p ≈ 2`
- Richardson: `O(h^4)`, theoretical slope `p ≈ 4`

Observed orders near the round-off-dominated region should not be interpreted as theoretical convergence orders.

## Round-off error

For sufficiently small `h`, the values `f(x+h)` and `f(x-h)` become extremely close. Their subtraction can suffer from cancellation, losing significant digits. Division by the small `2h` then amplifies the resulting floating-point error.

Therefore, decreasing `h` indefinitely does **not** guarantee a more accurate derivative. Each method has an empirically determined best `h` in the tested range.

## Project structure

```text
assignment3/
├── main.py
├── numerical_methods.py
├── functions.py
├── analysis.py
├── plotting.py
├── requirements.txt
├── README.md
└── output/
    ├── function_plots.png
    ├── central_difference_error.png
    ├── richardson_error.png
    ├── method_comparison.png
    ├── convergence_order.png
    ├── roundoff_behavior.png
    ├── results.csv
    ├── results.xlsx
    ├── results.md
    ├── best_h_summary.csv
    └── analysis.txt
```

## Installation

From the project directory:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Running `main.py` regenerates the complete `output/` directory.

## Generated files

- `function_plots.png` — all four test functions over `0 <= x <= 2`.
- `central_difference_error.png` — central-difference absolute-error convergence.
- `richardson_error.png` — Richardson absolute-error convergence.
- `method_comparison.png` — direct method comparison for each function.
- `convergence_order.png` — observed order and theoretical reference levels.
- `roundoff_behavior.png` — error behavior across the full small-`h` range.
- `results.csv` — complete numerical dataset.
- `results.xlsx` — formatted workbook with separate analysis sheets.
- `results.md` — complete human-readable Markdown table.
- `best_h_summary.csv` — best step size and minimum error for each function/method.
- `analysis.txt` — automatically generated evidence-based answers to Q1–Q6.
