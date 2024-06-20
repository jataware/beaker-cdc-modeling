# Description
Compute the Leslie matrix for a renewal process given the reproduction number and discrete generation interval probability mass vector.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike

def get_leslie_matrix(
    R: float, generation_interval_pmf: ArrayLike
) -> ArrayLike:
    """
    Create the Leslie matrix
    corresponding to a basic
    renewal process with the
    given R value and discrete
    generation interval pmf
    vector.

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    ArrayLike
        The Leslie matrix for the
        renewal process, as a jax array.
    """
    validate_discrete_dist_vector(generation_interval_pmf)
    gen_int_len = generation_interval_pmf.size
    aging_matrix = jnp.hstack(
        [
            jnp.identity(gen_int_len - 1),
            jnp.zeros(gen_int_len - 1)[..., jnp.newaxis],
        ]
    )


```
