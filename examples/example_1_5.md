# Description
Initializing the rest of the hospital admissions model components including infection seeding process, generation interval, Rt process, and observation model.

# Code
```
from pyrenew import model, process, observation, metaclass, deterministic, latent
import jax.numpy as jnp

# | label: initializing-rest-of-model
from pyrenew import model, process, observation, metaclass, transformation
from pyrenew.latent import InfectionSeedingProcess, SeedInfectionsExponential

# Infection process
latent_inf = latent.Infections()
I0 = InfectionSeedingProcess(
    "I0_seeding",
    metaclass.DistributionalRV(
        dist=dist.LogNormal(loc=jnp.log(100), scale=0.5), name="I0"
    ),
    SeedInfectionsExponential(
        gen_int_array.size,
        deterministic.DeterministicVariable(0.5, name="rate"),
    ),
)

# Generation interval and Rt
gen_int = deterministic.DeterministicPMF(gen_int, name="gen_int")
rtproc = process.RtRandomWalkProcess(
    Rt0_dist = dist.TruncatedNormal(loc=1.2, scale=0.2, low=0),
    Rt_transform = transformation.ExpTransform().inv,
    Rt_rw_dist = dist.Normal(0, 0.025),
)

# The observation model

```
