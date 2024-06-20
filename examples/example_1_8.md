# Description
Plotting Rt and observed hospital admissions for the basic renewal model.

# Code
```

# | label: fig-basic
# | fig-cap: Rt and Infections
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

# Rt plot
axs[0].plot(sim_data.Rt)
axs[0].set_ylabel("Rt")

# Infections plot
axs[1].plot(sim_data.observed_hosp_admissions)
axs[1].set_ylabel("Infections")
axs[1].set_yscale("log")

fig.suptitle("Basic renewal model")
fig.supxlabel("Time")
plt.tight_layout()

```
