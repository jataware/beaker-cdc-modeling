# Description
Calculate the asymptotic per-timestep growth rate and stable age distribution for a renewal process.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector

def get_asymptotic_growth_rate_and_age_dist(
    R: float, generation_interval_pmf: ArrayLike
) -> tuple[float, ArrayLike]:
    """
    Get the asymptotic per-timestep growth
    rate of the renewal process (the dominant
    eigenvalue of its Leslie matrix) and the
    associated stable age distribution
    (a normalized eigenvector associated to
    that eigenvalue).

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    tuple[float, ArrayLike]
        A tuple consisting of the asymptotic growth rate of
        the process, as jax float, and the stable age distribution
        of the process, as a jax array probability vector of the
        same shape as the generation interval probability vector.

    Raises
    ------
    ValueError
        If an age distribution vector with non-zero imaginary part is produced.
    """
    L = get_leslie_matrix(R, generation_interval_pmf)
    eigenvals, eigenvecs = jnp.linalg.eig(L)
    d = jnp.argmax(jnp.abs(eigenvals))  # index of dominant eigenvalue
    d_vec, d_val = eigenvecs[:, d], eigenvals[d]
    d_vec_real, d_val_real = jnp.real(d_vec), jnp.real(d_val)
    if not all(d_vec_real == d_vec):
        raise ValueError(
            "get_asymptotic_growth_rate_and_age_dist() "
            "produced an age distribution vector with "
            "non-zero imaginary part. "
            "Check your generation interval distribution "
            "vector and R value"
        )
    if not d_val_real == d_val:
        raise ValueError(
            "get_asymptotic_growth_rate_and_age_dist() "
            "produced an asymptotic growth rate with "
            "non-zero imaginary part. "
            "Check your generation interval distribution "
            "vector and R value"
        )
    d_vec_norm = d_vec_real / jnp.sum(d_vec_real)

```
