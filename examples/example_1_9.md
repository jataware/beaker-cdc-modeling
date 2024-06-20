# Description
Fitting the hospital admissions model to the data using MCMC sampling.

# Code
```

# | label: model-fit
import jax

hosp_model.run(
    num_samples=2000,
    num_warmup=2000,
    data_observed_hosp_admissions=dat["daily_hosp_admits"].to_numpy(),
    rng_key=jax.random.PRNGKey(54),
    mcmc_args=dict(progress_bar=False),

```
