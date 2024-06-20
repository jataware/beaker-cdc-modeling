# Description
Plotting daily hospital admissions from the simulated data.

# Code
```

# | label: fig-plot-hospital-admissions
# | fig-cap: Daily hospital admissions from the simulated data
import matplotlib.pyplot as plt

# Rotating the x-axis labels, and only showing ~10 labels
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
ax.xaxis.set_tick_params(rotation=45)
plt.plot(dat["date"].to_numpy(), dat["daily_hosp_admits"].to_numpy())
plt.xlabel("Date")
plt.ylabel("Admissions")

```
