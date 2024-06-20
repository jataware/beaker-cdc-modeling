# Description
Plotting the posterior distribution of hospital admissions with padding and day-of-the-week effects.

# Code
```

# | label: fig-output-admissions-padding-and-weekday
# | fig-cap: Hospital Admissions posterior distribution
out = hosp_model_weekday.plot_posterior(
    var="latent_hospital_admissions",
    ylab="Hospital Admissions",
    obs_signal=np.pad(
        dat_w_padding, (gen_int_array.size, 0), constant_values=np.nan
    ),

```
