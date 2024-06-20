# Description
Plotting the hospital admissions posterior distribution with padding.

# Code
```

# | label: fig-output-admissions-with-padding
# | fig-cap: Hospital Admissions posterior distribution
out = hosp_model.plot_posterior(
    var="latent_hospital_admissions",
    ylab="Hospital Admissions",
    obs_signal=np.pad(
        dat_w_padding, (gen_int_array.size, 0), constant_values=np.nan
    ),

```
