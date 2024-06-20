# Description
Sampling the periodic Rt with autoregressive difference using the RtPeriodicDiffProcess class.

# Code
```
from typing import NamedTuple
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.arrayutils import PeriodicBroadcaster
from pyrenew.metaclass import RandomVariable, _assert_sample_and_rtype
from pyrenew.process.firstdifferencear import FirstDifferenceARProcess

class RtPeriodicDiffProcessSample(NamedTuple):
    rt: ArrayLike | None = None

    def __repr__(self):

    def sample(
        self,
        duration: int,
        **kwargs,
    ) -> RtPeriodicDiffProcessSample:
        """
        Samples the periodic Rt with autoregressive difference.

        Parameters
        ----------
        duration : int
            Duration of the sequence.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        RtPeriodicDiffProcessSample
            Named tuple with "rt".
        """

        # Initial sample
        log_rt_prior = self.log_rt_prior.sample(**kwargs)[0]
        b = self.autoreg.sample(**kwargs)[0]
        s_r = self.periodic_diff_sd.sample(**kwargs)[0]

        # How many periods to sample?
        n_periods = int(jnp.ceil(duration / self.period_size))

        # Running the process
        ar_diff = FirstDifferenceARProcess(autoreg=b, noise_sd=s_r)
        log_rt = ar_diff.sample(
            duration=n_periods,
            init_val=log_rt_prior[1],
            init_rate_of_change=log_rt_prior[1] - log_rt_prior[0],
        )[0]

        return RtPeriodicDiffProcessSample(
            rt=self.broadcaster(jnp.exp(log_rt.flatten()), duration),

```
