# Description
Extracting and visualizing the generation interval and infection to hospitalization interval datasets.

# Code
```
from pyrenew import datasets

# | label: fig-data-extract
# | fig-cap: Generation interval and infection to hospitalization interval
gen_int = datasets.load_generation_interval()
inf_hosp_int = datasets.load_infection_admission_interval()

# We only need the probability_mass column of each dataset
gen_int_array = gen_int["probability_mass"].to_numpy()
gen_int = gen_int_array
inf_hosp_int = inf_hosp_int["probability_mass"].to_numpy()

# Taking a pick at the first 5 elements of each
gen_int[:5], inf_hosp_int[:5]

# Visualizing both quantities side by side
fig, axs = plt.subplots(1, 2)

axs[0].plot(gen_int)
axs[0].set_title("Generation interval")
axs[1].plot(inf_hosp_int)
axs[1].set_title("Infection to hospitalization interval")

```
