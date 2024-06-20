# Description
Generate infections according to a renewal process with a time-varying reproduction number R(t).

# Code
```
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.convolve import new_convolve_scanner

def compute_infections_from_rt(
    I0: ArrayLike,
    Rt: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
) -> ArrayLike:
    """
    Generate infections according to a
    renewal process with a time-varying
    reproduction number R(t)

    Parameters
    ----------
    I0 : ArrayLike
        Array of initial infections of the
        same length as the generation interval
        pmf vector.
    Rt : ArrayLike
        Timeseries of R(t) values
    reversed_generation_interval_pmf : ArrayLike
        discrete probability mass vector
        representing the generation interval
        of the infection process, where the final
        entry represents an infection 1 time unit in the
        past, the second-to-last entry represents
        an infection two time units in the past, etc.

    Returns
    -------
    ArrayLike
        The timeseries of infections, as a JAX array
    """
    incidence_func = new_convolve_scanner(
        reversed_generation_interval_pmf, IdentityTransform()
    )

    latest, all_infections = jax.lax.scan(f=incidence_func, init=I0, xs=Rt)


```
