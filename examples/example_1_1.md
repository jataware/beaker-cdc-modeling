# Description
Loading and preprocessing the wastewater dataset for hospital admissions analysis.

# Code
```
import polars as pl

# | label: data-inspect
import polars as pl
from pyrenew import datasets

dat = datasets.load_wastewater()
dat.head(5)
```

The data shows one entry per site, but the way it was simulated, the number of admissions is the same across sites. Thus, we will only keep the first observation per day.

```{python}
# | label: aggregation
# Keeping the first observation of each date
dat = dat.group_by("date").first().select(["date", "daily_hosp_admits"])

# Now, sorting by date
dat = dat.sort("date")

# Keeping the first 90 days
dat = dat.head(90)


```
