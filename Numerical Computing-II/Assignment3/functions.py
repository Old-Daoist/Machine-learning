"""Test functions and their exact analytical derivatives."""

import math

TEST_FUNCTIONS = [
    {
        "name": "exp(x)",
        "symbol": r"$f(x)=e^x$",
        "function": math.exp,
        "derivative": math.exp,
    },
    {
        "name": "sin(x)",
        "symbol": r"$f(x)=\sin(x)$",
        "function": math.sin,
        "derivative": math.cos,
    },
    {
        "name": "cos(x)",
        "symbol": r"$f(x)=\cos(x)$",
        "function": math.cos,
        "derivative": lambda x: -math.sin(x),
    },
    {
        "name": "x^3 - 2x + 1",
        "symbol": r"$f(x)=x^3-2x+1$",
        "function": lambda x: x**3 - 2.0*x + 1.0,
        "derivative": lambda x: 3.0*x**2 - 2.0,
    },
]
