# Description
Initialize the HospitalAdmissionsModel using previously defined initial conditions.

# Code
```

# Initializing the model
hospmodel = HospitalAdmissionsModel(
    gen_int_rv=gen_int,
    I0_rv=I0,
    latent_hosp_admissions_rv=latent_admissions,
    hosp_admission_obs_process_rv=admissions_process,
    latent_infections_rv=latent_infections,
    Rt_process_rv=Rt_process,

```
