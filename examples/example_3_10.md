# Description
Sample from the HospitalAdmissionsModel for 30 time steps and display the output.

# Code
```

with seed(rng_seed=np.random.randint(1, 60)):
    x = hospmodel.sample(n_timepoints_to_simulate=30)

```
