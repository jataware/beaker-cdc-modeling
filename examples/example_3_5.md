# Description
Example of initializing and visualizing a SimpleRandomWalkProcess.

# Code
```

# | label: fig-randwalk
# | fig-cap: Random walk example
np.random.seed(3312)
q = SimpleRandomWalkProcess(dist.Normal(0, 0.001))
with seed(rng_seed=np.random.randint(0, 1000)):
    q_samp = q.sample(n_timepoints=100)


```
