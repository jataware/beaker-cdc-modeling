# Description
Comparison simulation using the newly defined `InfFeedback` class with `RtInfectionsRenewalModel` and plotting the results.

# Code
```
import numpyro as npro
import jax.numpy as jnp
import numpy as np
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import InfectionSeedingProcess, SeedInfectionsExponential
from pyrenew.model import RtInfectionsRenewalModel
from pyrenew.process import RtRandomWalkProcess
import pyrenew.transformation as t
import numpyro.distributions as dist
import matplotlib.pyplot as plt

# Define and simulate with the first model
np.random.seed(223)
gen_int_array = jnp.array([0.25, 0.5, 0.15, 0.1])
gen_int = DeterministicPMF(gen_int_array, name="gen_int")
feedback_strength = DeterministicVariable(0.05, name="feedback_strength")
I0 = InfectionSeedingProcess(
    "I0_seeding",
    DistributionalRV(dist=dist.LogNormal(0, 1), name="I0"),
    SeedInfectionsExponential(
        gen_int_array.size,
        DeterministicVariable(0.5, name="rate"),
    ),
)
latent_infections = InfectionsWithFeedback(
    infection_feedback_strength=feedback_strength,
    infection_feedback_pmf=gen_int,
)
rt = RtRandomWalkProcess(
    Rt0_dist = dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
    Rt_transform = t.ExpTransform().inv,
    Rt_rw_dist = dist.Normal(0, 0.025),
)

# Build the first model
model0 = RtInfectionsRenewalModel(
    gen_int_rv=gen_int,
    I0_rv=I0,
    latent_infections_rv=latent_infections,
    Rt_process_rv=rt,
    infection_obs_process_rv=None,
)

# Simulate the first model
with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
    model0_samp = model0.sample(n_timepoints_to_simulate=30)

# Define the new latent infections using InfFeedback
latent_infections2 = InfFeedback(
    infection_feedback_strength=feedback_strength,
    infection_feedback_pmf=gen_int,
)

# Build the second model
model1 = RtInfectionsRenewalModel(
    gen_int_rv=gen_int,
    I0_rv=I0,
    latent_infections_rv=latent_infections2,
    Rt_process_rv=rt,
    infection_obs_process_rv=None,
)

# Simulate the second model
with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):

# | label: simulation2
latent_infections2 = InfFeedback(
    infection_feedback_strength=feedback_strength,
    infection_feedback_pmf=gen_int,
)

model1 = RtInfectionsRenewalModel(
    gen_int_rv=gen_int,
    I0_rv=I0,
    latent_infections_rv=latent_infections2,
    Rt_process_rv=rt,
    infection_obs_process_rv=None,
)

# Sampling and fitting model 0 (with no obs for infections)
np.random.seed(223)
with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
    model1_samp = model1.sample(n_timepoints_to_simulate=30)
```

Comparing `model0` with `model1`, these two should match:

```{python}
# | label: fig-model0-vs-model1
# | fig-cap: Comparing latent infections from model 0 and model 1
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2)
ax[0].plot(model0_samp.latent_infections)
ax[1].plot(model1_samp.latent_infections)
ax[0].set_xlabel("Time (model 0)")
ax[1].set_xlabel("Time (model 1)")
ax[0].set_ylabel("Infections")

```
