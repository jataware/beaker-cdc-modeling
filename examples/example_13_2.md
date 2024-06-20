# Description
Example of creating seed infections from a vector of infections with the same length as the number of timepoints.

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

class SeedInfectionsFromVec(InfectionSeedMethod):
    """Create seed infections from a vector of infections."""

    def seed_infections(self, I_pre_seed: ArrayLike):
        """Create seed infections from a vector of infections.

        Parameters
        ----------
        I_pre_seed : ArrayLike
            An array with the same length as ``n_timepoints`` to be used as the seed infections.

        Returns
        -------
        ArrayLike
            An array of length ``n_timepoints`` with the number of seeded infections at each time point.
        """
        if I_pre_seed.size != self.n_timepoints:
            raise ValueError(
                "I_pre_seed must have the same size as n_timepoints. "
                f"Got I_pre_seed of size {I_pre_seed.size} "
                f"and n_timepoints of size {self.n_timepoints}."
            )

```
