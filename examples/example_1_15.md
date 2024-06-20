# Description
Rebuilding the latent hospitalization model to incorporate day-of-the-week effects.

# Code
```

# | label: latent-hosp-weekday
latent_hosp_wday_effect = latent.HospitalAdmissions(
    infection_to_admission_interval_rv=inf_hosp_int,
    infect_hosp_rate_rv=hosp_rate,
    day_of_week_effect_rv=dayofweek_effect,
)

hosp_model_weekday = model.HospitalAdmissionsModel(
    latent_infections_rv=latent_inf,
    latent_hosp_admissions_rv=latent_hosp_wday_effect,
    I0_rv=I0,
    gen_int_rv=gen_int,
    Rt_process_rv=rtproc,
    hosp_admission_obs_process_rv=obs,

```
