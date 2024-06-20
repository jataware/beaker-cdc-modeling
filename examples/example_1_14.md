# Description
Defining a custom day-of-the-week effect model as a random variable using a truncated normal distribution.

# Code
```
from pyrenew import metaclass
import numpyro as npro

# | label: weekly-effect
from pyrenew import metaclass
import numpyro as npro

class DayOfWeekEffect(metaclass.RandomVariable):
    """Day of the week effect"""

    def __init__(self, len: int):
        """Initialize the day of the week effect distribution
        Parameters
        ----------
        len : int
            The number of observations
        """
        self.nweeks = int(jnp.ceil(len / 7))
        self.len = len

    @staticmethod
    def validate():
        return None

    def sample(self, **kwargs):
        ans = npro.sample(
            name="dayofweek_effect",
            fn=npro.distributions.TruncatedNormal(
                loc=1.0, scale=0.5, low=0.1, high=10.0
            ),
            sample_shape=(7,),
        )

        return jnp.tile(ans, self.nweeks)[: self.len]


# Initializing the RV

```
