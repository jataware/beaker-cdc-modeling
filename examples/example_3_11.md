# Description
Visualize the results of the HospitalAdmissionsModel for infections, latent hospital admissions, and observed hospital admissions.

# Code
```

# | label: fig-hosp
# | fig-cap: Infections
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(x.latent_infections)
ax[0].set_ylim([1 / 5, 5])
ax[1].plot(x.latent_hosp_admissions)
ax[2].plot(x.observed_hosp_admissions, "o")
for axis in ax[:-1]:

```
