# Description
Visualizing the posterior distribution of hospital admissions after model fitting.

# Code
```

# | label: fig-output-hospital-admissions
# | fig-cap: Hospital Admissions posterior distribution
out = hosp_model.plot_posterior(
    var="latent_hospital_admissions",
    ylab="Hospital Admissions",
    obs_signal=np.pad(
        dat["daily_hosp_admits"].to_numpy().astype(float),
        (gen_int_array.size, 0),
        constant_values=np.nan,
    ),

```
