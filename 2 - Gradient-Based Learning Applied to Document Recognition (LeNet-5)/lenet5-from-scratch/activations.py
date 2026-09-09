"""
activations.py — Activation Functions for LeNet-5

From the paper (Section II-B):
  f(a) = A * tanh(S * a)
  where A = 1.7159, S = 2/3

This specific scaling is chosen so that:
  - f(±1) ≈ ±1  (nice, symmetric output range)
  - The second derivative is maximized at x = ±1, meaning learning
    speed is highest when neuron outputs are near ±1

The derivative is:
  f'(a) = A * S * (1 - tanh²(S * a))
       = (S / A) * (A - f(a)) * (A + f(a))
       = (S / A) * (A² - f(a)²)

The second form is more efficient because we can reuse f(a) from
the forward pass instead of recomputing tanh.
"""

import numpy as np

# Constants from the paper
A = 1.7159
S = 2.0 / 3.0


def scaled_tanh(x):
    """
    Forward pass of the scaled hyperbolic tangent activation.

    f(x) = 1.7159 * tanh(2/3 * x)

    Parameters
    ----------
    x : np.ndarray
        Pre-activation values (any shape).

    Returns
    -------
    np.ndarray
        Activated values, same shape as x.
        Output range: approximately (-1.7159, +1.7159)

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> scaled_tanh(x)
    array([-1.4503, -1.0487, 0.0, 1.0487, 1.4503])  # approximately
    """
    return A * np.tanh(S * x)


def scaled_tanh_derivative(output):
    """
    Derivative of scaled tanh, computed from the OUTPUT (not input).

    f'(a) = (S / A) * (A² - f(a)²)

    This is the efficient form: we pass in f(a) (the output from the
    forward pass) and avoid recomputing tanh.

    Parameters
    ----------
    output : np.ndarray
        The OUTPUT of scaled_tanh (i.e., f(a)), not the input a.

    Returns
    -------
    np.ndarray
        The derivative f'(a) at each point, same shape as output.

    Why use output instead of input?
    --------------------------------
    During backpropagation, we already have the forward pass output
    stored. Computing the derivative from the output is:
      1. Faster (no tanh recomputation)
      2. More memory-efficient (we already store the output)

    Example
    -------
    >>> output = scaled_tanh(np.array([0.0]))
    >>> scaled_tanh_derivative(output)
    array([1.1439])  # approximately A * S = 1.7159 * 2/3
    """
    return (S / A) * (A * A - output * output)
