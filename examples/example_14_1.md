# Description
Sampling infections given reproduction number, initial infections, and generation interval using the Infections class.

# Code
```
from typing import NamedTuple
import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

class InfectionsSample(NamedTuple):
    infections: ArrayLike | None = None

class Infections(RandomVariable):
    def __init__(self, infections_mean_varname: str = "latent_infections") -> None:
        self.infections_mean_varname = infections_mean_varname
        return None
    

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> InfectionsSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections vector
            of the same length as the
            generation interval.
        gen_int : ArrayLike
            Generation interval pmf vector.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """
        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size}, "
                f"generation interval length: {gen_int.size}."
            )

        gen_int_rev = jnp.flip(gen_int)
        recent_I0 = I0[-gen_int_rev.size :]

        all_infections = inf.compute_infections_from_rt(
            I0=recent_I0,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        all_infections = jnp.hstack([I0, all_infections])

        npro.deterministic(self.infections_mean_varname, all_infections)


```
