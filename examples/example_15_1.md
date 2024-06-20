# Description
Example demonstrating how to sample infections given Rt, initial infections, and generation interval using the `InfectionsWithFeedback` class.

# Code
```
import jax.numpy as jnp
import numpyro as npro
import pyrenew.arrayutils as au
import pyrenew.latent.infection_functions as inf
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable, _assert_sample_and_rtype

class InfectionsRtFeedbackSample(NamedTuple):
    infections: ArrayLike | None = None
    rt: ArrayLike | None = None

class InfectionsWithFeedback(RandomVariable):
    def __init__(
        self,
        infection_feedback_strength: RandomVariable,
        infection_feedback_pmf: RandomVariable,
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        self.validate(infection_feedback_strength, infection_feedback_pmf)
        self.infection_feedback_strength = infection_feedback_strength
        self.infection_feedback_pmf = infection_feedback_pmf
        self.infections_mean_varname = infections_mean_varname
        return None

    @staticmethod
    def validate(
        inf_feedback_strength: any,
        inf_feedback_pmf: any,
    ) -> None:
        _assert_sample_and_rtype(inf_feedback_strength)
        _assert_sample_and_rtype(inf_feedback_pmf)
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> InfectionsRtFeedbackSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections, as an array
            at least as long as the generation
            interval PMF.
        gen_int : ArrayLike
            Generation interval PMF.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsWithFeedback
            Named tuple with "infections".
        """
        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections must be at least as long as the "
                f"generation interval. Got {I0.size} initial infections "
                f"and {gen_int.size} generation interval."
            )

        gen_int_rev = jnp.flip(gen_int)

        I0 = I0[-gen_int_rev.size :]

        # Sampling inf feedback strength
        inf_feedback_strength, *_ = self.infection_feedback_strength.sample(
            **kwargs,
        )

        # Making sure inf_feedback_strength spans the Rt length
        if inf_feedback_strength.size == 1:
            inf_feedback_strength = au.pad_x_to_match_y(
                x=inf_feedback_strength,
                y=Rt,
                fill_value=inf_feedback_strength[0],
            )
        elif inf_feedback_strength.size != Rt.size:
            raise ValueError(
                "Infection feedback strength must be of size 1 or the same "
                f"size as the reproduction number. Got {inf_feedback_strength.size} "
                f"and {Rt.size} respectively."
            )

        # Sampling inf feedback pmf
        inf_feedback_pmf, *_ = self.infection_feedback_pmf.sample(**kwargs)

        inf_fb_pmf_rev = jnp.flip(inf_feedback_pmf)

        all_infections, Rt_adj = inf.compute_infections_from_rt_with_feedback(
            I0=I0,
            Rt_raw=Rt,
            infection_feedback_strength=inf_feedback_strength,
            reversed_generation_interval_pmf=gen_int_rev,
            reversed_infection_feedback_pmf=inf_fb_pmf_rev,
        )

        # Appending initial infections to the infections
        all_infections = jnp.hstack([I0, all_infections])

        npro.deterministic("Rt_adjusted", Rt_adj)

        return InfectionsRtFeedbackSample(
            infections=all_infections,
            rt=Rt_adj,

```
