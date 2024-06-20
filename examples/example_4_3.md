# Description
Obtain the stable age distribution for a renewal process using the reproduction number and discrete generation interval probability mass vector.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.distutil import validate_discrete_dist_vector

def get_stable_age_distribution(
    R: float, generation_interval_pmf: ArrayLike
) -> ArrayLike:
    """
    Get the stable age distribution for a
    renewal process with a given value of
    R and a given discrete generation
    interval probability mass vector.

    This function computes that stable age
    distribution by finding and then normalizing
    an eigenvector associated to the dominant
    eigenvalue of the renewal process's
    Leslie matrix.

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
        The stable age distribution for the
        process, as a jax array probability vector of
        the same shape as the generation interval
        probability vector.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[
        1

```
