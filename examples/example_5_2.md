# Description
Reverse a discrete distribution vector, useful for discrete time-to-event distributions.

# Code
```
import jax.numpy as jnp

def reverse_discrete_dist_vector(dist: ArrayLike) -> ArrayLike:
    """
    Reverse a discrete distribution
    vector (useful for discrete
    time-to-event distributions).

    Parameters
    ----------
    dist : ArrayLike
        A discrete distribution vector (likely discrete time-to-event distribution)

    Returns
    -------
    ArrayLike
        A reversed (jnp.flip) discrete distribution vector
    """

```
