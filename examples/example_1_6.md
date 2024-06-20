# Description
Building the complete hospital admissions model.

# Code
```

# | label: init-model
hosp_model = model.HospitalAdmissionsModel(
    latent_infections_rv=latent_inf,
    latent_hosp_admissions_rv=latent_hosp,
    I0_rv=I0,
    gen_int_rv=gen_int,
    Rt_process_rv=rtproc,
    hosp_admission_obs_process_rv=obs,

```
