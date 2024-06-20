# Description
Sample from the negative binomial distribution using a custom negative binomial observation class.

# Code
```
from __future__ import annotations
import numbers as nums
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

class NegativeBinomialObservation(RandomVariable):
    """Negative Binomial observation"""

    def __init__(
        self,
        concentration_prior: dist.Distribution | ArrayLike,
        concentration_suffix: str | None = "_concentration",
        parameter_name="negbinom_rv",
        eps: float = 1e-10,
    ) -> None:
        NegativeBinomialObservation.validate(concentration_prior)

        if isinstance(concentration_prior, dist.Distribution):
            self.sample_prior = lambda: numpyro.sample(
                self.parameter_name + self.concentration_suffix,
                concentration_prior,
            )
        else:
            self.sample_prior = lambda: concentration_prior

        self.parameter_name = parameter_name
        self.concentration_suffix = concentration_suffix
        self.eps = eps

    @staticmethod
    def validate(concentration_prior: any) -> None:
        assert isinstance(
            concentration_prior, (dist.Distribution, nums.Number)
        )

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sample from the negative binomial distribution

        Parameters
        ----------
        mu : ArrayLike
            Mean parameter of the negative binomial distribution.
        obs : ArrayLike, optional
            Observed data, by default None.
        name : str, optional
            Name of the random variable if other than that defined during
            construction, by default None (self.parameter_name).
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
        """
        concentration = self.sample_prior()

        if name is None:
            name = self.parameter_name

        return (
            numpyro.sample(
                name=name,
                fn=dist.NegativeBinomial2(
                    mean=mu + self.eps,
                    concentration=concentration,
                ),
                obs=obs,
            ),

```
