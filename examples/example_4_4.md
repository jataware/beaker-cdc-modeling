# Description
Get the asymptotic per-timestep growth rate for a renewal process.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector

def get_asymptotic_growth_rate(
    R: float, generation_interval_pmf: ArrayLike
) -> float:
    """
    Get the asymptotic per timestep growth rate
    for a renewal process with a given value of
    R and a given discrete generation interval
    probability mass vector.

    This function computes that growth rate
    finding the dominant eigenvalue of the
    renewal process's Leslie matrix.

    Parameters
    ----------
    R : float
       The reproduction number of the renewal process
    generation_interval_pmf: ArrayLike
       The discrete generation interval probability
       mass vector of the renewal process

    Returns
    -------
    float
        The asymptotic growth rate of the renewal process,
        as a jax float.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[
        0

```
