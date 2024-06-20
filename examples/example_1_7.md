# Description
Simulating to check if the hospital admissions model is working.

# Code
```
import numpyro as npro

# | label: simulation
import numpyro as npro
import numpy as np

timeframe = 120

np.random.seed(223)
with npro.handlers.seed(rng_seed=np.random.randint(1, timeframe)):

```
