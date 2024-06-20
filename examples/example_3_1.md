# Description
Factory function to create a "scanner" function for single-step convolution using jax.lax.scan.

# Code
```
from __future__ import annotations

from typing import Callable
import jax.numpy as jnp

def new_convolve_scanner(
    array_to_convolve: ArrayLike,
    transform: Callable,
) -> Callable:
    r"""
    Factory function to create a "scanner" function
    that can be used with :py:func:`jax.lax.scan` to
    construct an array via backward-looking iterative
    convolution.

    Parameters
    ----------
    array_to_convolve : ArrayLike
        A 1D jax array to convolve with subsets of the
        iteratively constructed history array.

    transform : Callable
        A transformation to apply to the result
        of the dot product and multiplication.

    Returns
    -------
    Callable
        A scanner function that can be used with
        :py:func:`jax.lax.scan` for convolution.
        This function takes a history subset array and
        a scalar, computes the dot product of
        the supplied convolution array with the history
        subset array, multiplies by the scalar, and
        returns the resulting value and a new history subset
        array formed by the 2nd-through-last entries
        of the old history subset array followed by that same
        resulting value.

    Notes
    -----
    The following iterative operation is found often
    in renewal processes:

    .. math::
        X(t) = f\left(m(t) \begin{bmatrix} X(t - n) \\ X(t - n + 1) \\
        \vdots{} \\ X(t - 1)\end{bmatrix} \cdot{} \mathbf{d} \right)

    Where :math:`\mathbf{d}` is a vector of length :math:`n`,
    :math:`m(t)` is a scalar for each value of time :math:`t`,
    and :math:`f` is a scalar-valued function.

    Given :math:`\mathbf{d}`, and optionally :math:`f`,
    this factory function returns a new function that
    peforms one step of this process while scanning along
    an array of  multipliers (i.e. an array
    giving the values of :math:`m(t)`) using :py:func:`jax.lax.scan`.
    """

    def _new_scanner(
        history_subset: ArrayLike, multiplier: float
    ) -> tuple[ArrayLike, float]:  # numpydoc ignore=GL08
        new_val = transform(
            multiplier * jnp.dot(array_to_convolve, history_subset)
        )
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, new_val


```
