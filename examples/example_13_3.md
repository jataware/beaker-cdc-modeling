# Description
Example of generating seed infections according to exponential growth.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike
from abc import ABCMeta, abstractmethod
from pyrenew.metaclass import RandomVariable

class InfectionSeedMethod(metaclass=ABCMeta):
    """Method for seeding initial infections in a renewal process."""

    def __init__(self, n_timepoints: int):
        self.validate(n_timepoints)
        self.n_timepoints = n_timepoints

    @staticmethod
    def validate(n_timepoints: int) -> None:
        if not isinstance(n_timepoints, int):
            raise TypeError(
                f"n_timepoints must be an integer. Got {type(n_timepoints)}"
            )
        if n_timepoints <= 0:
            raise ValueError(
                f"n_timepoints must be positive. Got {n_timepoints}"
            )

    @abstractmethod
    def seed_infections(self, I_pre_seed: ArrayLike):
        pass

    def __call__(self, I_pre_seed: ArrayLike):

class SeedInfectionsExponential(InfectionSeedMethod):
    r"""Generate seed infections according to exponential growth.

    Notes
    -----
    The number of incident infections at time `t` is given by:

    .. math:: I(t) = I_p \exp \left( r (t - t_p) \right)

    Where :math:`I_p` is ``I_pre_seed``, :math:`r` is ``rate``, and :math:`t_p` is ``t_pre_seed``.
    This ensures that :math:`I(t_p) = I_p`.
    We default to ``t_pre_seed = n_timepoints - 1``, so that
    ``I_pre_seed`` represents the number of incident infections immediately
    before the renewal process begins.
    """

    def __init__(
        self,
        n_timepoints: int,
        rate: RandomVariable,
        t_pre_seed: int | None = None,
    ):
        """Default constructor for the ``SeedInfectionsExponential`` class.

        Parameters
        ----------
        n_timepoints : int
            the number of time points to generate seed infections for
        rate : RandomVariable
            A random variable representing the rate of exponential growth
        t_pre_seed : int | None, optional
             The time point whose number of infections is described by ``I_pre_seed``. Defaults to ``n_timepoints - 1``.
        """
        super().__init__(n_timepoints)
        self.rate = rate
        if t_pre_seed is None:
            t_pre_seed = n_timepoints - 1
        self.t_pre_seed = t_pre_seed

    def seed_infections(self, I_pre_seed: ArrayLike):
        """Generate seed infections according to exponential growth.

        Parameters
        ----------
        I_pre_seed : ArrayLike
            An array of size 1 representing the number of infections at time ``t_pre_seed``.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of seeded infections at each time point.
        """
        if I_pre_seed.size != 1:
            raise ValueError(
                f"I_pre_seed must be an array of size 1. Got size {I_pre_seed.size}."
            )
        (rate,) = self.rate.sample()
        if rate.size != 1:
            raise ValueError(
                f"rate must be an array of size 1. Got size {rate.size}."
            )
        return I_pre_seed * jnp.exp(

```
