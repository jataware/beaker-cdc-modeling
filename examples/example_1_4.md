# Description
Defining the latent hospital admissions model with infection to hospitalization interval and infection to hospitalization rate.

# Code
```
from pyrenew import latent, deterministic, metaclass
import jax.numpy as jnp

# | label: latent-hosp
from pyrenew import latent, deterministic, metaclass
import jax.numpy as jnp
import numpyro.distributions as dist

inf_hosp_int = deterministic.DeterministicPMF(
    inf_hosp_int, name="inf_hosp_int"
)

hosp_rate = metaclass.DistributionalRV(
    dist=dist.LogNormal(jnp.log(0.05), 0.1),
    name="IHR",
)

latent_hosp = latent.HospitalAdmissions(
    infection_to_admission_interval_rv=inf_hosp_int,
    infect_hosp_rate_rv=hosp_rate,

```
