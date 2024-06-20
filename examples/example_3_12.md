# Description
Fit the HospitalAdmissionsModel to the simulated data using MCMC.

# Code
```

hospmodel.run(
    num_warmup=1000,
    num_samples=1000,
    data_observed_hosp_admissions=x.observed_hosp_admissions,
    rng_key=jax.random.PRNGKey(54),
    mcmc_args=dict(progress_bar=False),

```
