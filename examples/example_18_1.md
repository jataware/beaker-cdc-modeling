# Description
Sampling from an AR(p) process in Numpyro using a custom object.

# Code
```
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

class ARProcess(RandomVariable):
    def __init__(self, mean: float, autoreg: ArrayLike, noise_sd: float) -> None:
        self.mean = mean
        self.autoreg = autoreg

    def sample(
        self,
        duration: int,
        inits: ArrayLike = None,
        name: str = "arprocess",
        **kwargs,
    ) -> tuple:
        """
        Sample from the AR process

        Parameters
        ----------
        duration: int
            Length of the sequence.
        inits : ArrayLike, optional
            Initial points, if None, then these are sampled.
            Defaults to None.
        name : str, optional
            Name of the parameter passed to numpyro.sample.
            Defaults to "arprocess".
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (duration,).
        """
        order = self.autoreg.shape[0]
        if inits is None:
            inits = numpyro.sample(
                name + "_sampled_inits",
                dist.Normal(0, self.noise_sd).expand((order,)),
            )

        def _ar_scanner(carry, next):  # numpydoc ignore=GL08
            new_term = (jnp.dot(self.autoreg, carry) + next).flatten()
            new_carry = jnp.hstack([new_term, carry[: (order - 1)]])
            return new_carry, new_term

        noise = numpyro.sample(
            name + "_noise",
            dist.Normal(0, self.noise_sd).expand((duration - inits.size,)),
        )

        last, ts = lax.scan(_ar_scanner, inits - self.mean, noise)

```
