# Description
Implementation of the `InfectionsWithFeedback` class, which includes the `sample` method using `compute_infections_from_rt_with_feedback` and ensuring vector lengths with `pad_x_to_match_y`.

# Code
```
from pyrenew.metaclass import RandomVariable
from pyrenew.latent import compute_infections_from_rt_with_feedback
from pyrenew import arrayutils as au
from jax.typing import ArrayLike
import jax.numpy as jnp

InfFeedbackSample = namedtuple(
    typename="InfFeedbackSample",
    field_names=["infections", "rt"],
    defaults=(None, None),

# | label: new-model-def
# | code-line-numbers: true
# Creating the class
from pyrenew.metaclass import RandomVariable
from pyrenew.latent import compute_infections_from_rt_with_feedback
from pyrenew import arrayutils as au
from jax.typing import ArrayLike
import jax.numpy as jnp


class InfFeedback(RandomVariable):
    """Latent infections"""

    def __init__(
        self,
        infection_feedback_strength: RandomVariable,
        infection_feedback_pmf: RandomVariable,
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        """Constructor"""

        self.infection_feedback_strength = infection_feedback_strength
        self.infection_feedback_pmf = infection_feedback_pmf
        self.infections_mean_varname = infections_mean_varname

        return None

    def validate(self):
        """
        Generally, this method should be more meaningful, but we will skip it for now
        """
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> tuple:
        """Sample infections with feedback"""

        # Generation interval
        gen_int_rev = jnp.flip(gen_int)

        # Baseline infections
        I0_vec = I0[-gen_int_rev.size :]

        # Sampling inf feedback strength and adjusting the shape
        inf_feedback_strength, *_ = self.infection_feedback_strength.sample(
            **kwargs,
        )
        inf_feedback_strength = au.pad_x_to_match_y(
            x=inf_feedback_strength, y=Rt, fill_value=inf_feedback_strength[0]
        )

        # Sampling inf feedback and adjusting the shape
        inf_feedback_pmf, *_ = self.infection_feedback_pmf.sample(**kwargs)
        inf_fb_pmf_rev = jnp.flip(inf_feedback_pmf)

        # Generating the infections with feedback
        all_infections, Rt_adj = compute_infections_from_rt_with_feedback(
            I0=I0_vec,
            Rt_raw=Rt,
            infection_feedback_strength=inf_feedback_strength,
            reversed_generation_interval_pmf=gen_int_rev,
            reversed_infection_feedback_pmf=inf_fb_pmf_rev,
        )

        # Storing adjusted Rt for future use
        npro.deterministic("Rt_adjusted", Rt_adj)

        # Preparing theoutput

        return InfFeedbackSample(
            infections=all_infections,
            rt=Rt_adj,

```
