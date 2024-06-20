# Description
Visualize the posterior Rt samples using spread_draws function output.

# Code
```

# | label: fig-sampled-rt
# | fig-cap: Posterior Rt
import numpy as np
import polars as pl

fig, ax = plt.subplots(figsize=[4, 5])

ax.plot(x[0])
samp_ids = np.random.randint(size=25, low=0, high=999)
for samp_id in samp_ids:
    sub_samps = samps.filter(pl.col("draw") == samp_id).sort(pl.col("time"))
    ax.plot(
        sub_samps.select("time").to_numpy(),
        sub_samps.select("Rt").to_numpy(),
        color="darkblue",
        alpha=0.1,
    )
ax.set_ylim([0.4, 1 / 0.4])
ax.set_yticks([0.5, 1, 2])

```
