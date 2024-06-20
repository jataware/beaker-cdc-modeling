# Description
Demonstration of sampling from a custom Markovian random walk process with an arbitrary step distribution using NumPyro and JAX.

# Code
```
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist

class SimpleRandomWalkProcess(RandomVariable):
    def __init__(self, error_distribution: dist.Distribution) -> None:
        self.error_distribution = error_distribution

    @staticmethod
    def validate():

    def sample(
        self,
        n_timepoints: int,
        name: str = "randomwalk",
        init: float = None,
        **kwargs,
    ) -> tuple:
        """
        Samples from the randomwalk

        Parameters
        ----------
        n_timepoints : int
            Length of the walk.
        name : str, optional
            Passed to numpyro.sample, by default "randomwalk"
        init : float, optional
            Initial point of the walk, by default None
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_timepoints,).
        """

        if init is None:
            init = npro.sample(name + "_init", self.error_distribution)
        diffs = npro.sample(
            name + "_diffs",
            self.error_distribution.expand((n_timepoints - 1,)),
        )


```
