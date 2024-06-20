# Description
Running the redefined hospital admissions model with day-of-the-week effects.

# Code
```

# | label: model-2-run
hosp_model_weekday.run(
    num_samples=2000,
    num_warmup=2000,
    data_observed_hosp_admissions=dat_w_padding,
    rng_key=jax.random.PRNGKey(54),
    mcmc_args=dict(progress_bar=False),
    padding=days_to_impute,

```
