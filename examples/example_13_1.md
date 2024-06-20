# Description
Example of creating a seed infection vector by padding a shorter vector with zeros at the beginning of the time series.

# Code
```
import jax.numpy as jnp
from jax.typing import ArrayLike
from abc import ABCMeta, abstractmethod

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

class SeedInfectionsZeroPad(InfectionSeedMethod):
    """
    Create a seed infection vector of specified length by
    padding a shorter vector with an appropriate number of
    zeros at the beginning of the time series.
    """

    def seed_infections(self, I_pre_seed: ArrayLike):
        """Pad the seed infections with zeros at the beginning of the time series.

        Parameters
        ----------
        I_pre_seed : ArrayLike
            An array with seeded infections to be padded with zeros.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of seeded infections at each time point.
        """
        if self.n_timepoints < I_pre_seed.size:
            raise ValueError(
                "I_pre_seed must be no longer than n_timepoints. "
                f"Got I_pre_seed of size {I_pre_seed.size} and "
                f" n_timepoints of size {self.n_timepoints}."
            )

```
