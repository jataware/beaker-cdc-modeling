# Description
An example of generating samples from a Rt Randomwalk Process using the RtRandomWalkProcess class.

# Code
```
import numpyro as npro
import numpyro.distributions as dist
import pyrenew.transformation as t
from pyrenew.metaclass import RandomVariable

    def sample(
        self,
        n_timepoints: int,
        **kwargs,
    ) -> tuple:
        """
        Generate samples from the process

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_timepoints,).
        """

        Rt0 = npro.sample("Rt0", self.Rt0_dist)

        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalkProcess(self.Rt_rw_dist)
        Rt_trans_ts, *_ = Rt_trans_proc.sample(
            n_timepoints=n_timepoints,
            name="Rt_transformed_rw",
            init=Rt0_trans,
        )

        Rt = npro.deterministic("Rt", self.Rt_transform.inv(Rt_trans_ts))


```
