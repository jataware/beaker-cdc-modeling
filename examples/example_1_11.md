# Description
Padding the model to account for initial observation issues and re-estimating with 21 days of imputed data.

# Code
```

# | label: model-fit-padding
days_to_impute = 21

# Add 21 Nas to the beginning of dat_w_padding
dat_w_padding = np.pad(
    dat["daily_hosp_admits"].to_numpy().astype(float),
    (days_to_impute, 0),
    constant_values=np.nan,
)


hosp_model.run(
    num_samples=2000,
    num_warmup=2000,
    data_observed_hosp_admissions=dat_w_padding,
    rng_key=jax.random.PRNGKey(54),
    mcmc_args=dict(progress_bar=False),
    padding=days_to_impute,  # Padding the model

```
