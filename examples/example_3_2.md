# Description
Factory function to create a scanner function for double-step convolution using jax.lax.scan.

# Code
```
from __future__ import annotations

from typing import Callable
import jax.numpy as jnp

def new_double_convolve_scanner(
    arrays_to_convolve: tuple[ArrayLike, ArrayLike],
    transforms: tuple[Callable, Callable],
) -> Callable:
    r"""
    Factory function to create a scanner function
    that iteratively constructs arrays by applying
    the dot-product/multiply/transform operation
    twice per history subset, with the first yielding
    operation yielding an additional scalar multiplier
    for the second.

    Parameters
    ----------
    arrays_to_convolve : tuple[ArrayLike, ArrayLike]
        A tuple of two 1D jax arrays, one for
        each of the two stages of convolution.
        The first entry in the arrays_to_convolve
        tuple will be convolved with the
        current history subset array first, the
        the second entry will be convolved with
        it second.
    transforms : tuple[Callable, Callable]
        A tuple of two functions, each transforming the
        output of the dot product at each
        convolution stage. The first entry in the transforms
        tuple will be applied first, then the second will
        be applied.

    Returns
    -------
    Callable
        A scanner function that applies two sets of
        convolution, multiply, and transform operations
        in sequence to construct a new array by scanning
        along a pair of input arrays that are equal in
        length to each other.

    Notes
    -----
    Using the same notation as in the documentation for
    :func:`new_convolve_scanner`, this function aids in
    applying the iterative operation:

    .. math::
        \begin{aligned}
        Y(t) &= f_1 \left(m_1(t)
           \begin{bmatrix}
                X(t - n) \\
                X(t - n + 1) \\
                \vdots{} \\
                X(t - 1)
        \end{bmatrix} \cdot{} \mathbf{d}_1 \right) \\ \\
        X(t) &= f_2 \left(
           m_2(t) Y(t)
        \begin{bmatrix} X(t - n) \\ X(t - n + 1) \\
        \vdots{} \\ X(t - 1)\end{bmatrix} \cdot{} \mathbf{d}_2 \right)
        \end{aligned}

    Where :math:`\mathbf{d}_1` and :math:`\mathbf{d}_2` are vectors of
    length :math:`n`, :math:`m_1(t)` and :math:`m_2(t)` are scalars
    for each value of time :math:`t`, and :math:`f_1` and :math:`f_2`
    are scalar-valued functions.
    """
    arr1, arr2 = arrays_to_convolve
    t1, t2 = transforms

    def _new_scanner(
        history_subset: ArrayLike,
        multipliers: tuple[float, float],
    ) -> tuple[ArrayLike, tuple[float, float]]:  # numpydoc ignore=GL08
        m1, m2 = multipliers
        m_net1 = t1(m1 * jnp.dot(arr1, history_subset))
        new_val = t2(m2 * m_net1 * jnp.dot(arr2, history_subset))
        latest = jnp.hstack([history_subset[1:], new_val])
        return latest, (new_val, m_net1)


```
