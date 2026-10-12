"""Numerical differentiation methods used in Assignment 3."""

def central_difference(f, x, h):
    """Second-order central-difference approximation."""
    return (f(x + h) - f(x - h)) / (2.0 * h)

def richardson_extrapolation(f, x, h):
    """Fourth-order Richardson extrapolation from central differences."""
    d_h = central_difference(f, x, h)
    d_half = central_difference(f, x, h / 2.0)
    return (4.0 * d_half - d_h) / 3.0
