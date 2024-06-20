# Description
Example demonstrating how to sample from a Poisson process using numpyro's sample function.

# Code
```
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

class PoissonObservation(RandomVariable):
    def __init__(self, parameter_name: str = "poisson_rv", eps: float = 1e-8) -> None:
        self.parameter_name = parameter_name
        self.eps = eps

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the Poisson process

        Parameters
        ----------
        mu : ArrayLike
            Rate parameter of the Poisson distribution.
        obs : ArrayLike | None, optional
            Observed data. Defaults to None.
        name : str | None, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """

        if name is None:
            name = self.parameter_name

        return (
            numpyro.sample(
                name=name,
                fn=dist.Poisson(rate=mu + self.eps),
                obs=obs,
            ),

```
