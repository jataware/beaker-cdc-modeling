# Description
Sampling from the Basic Renewal Model (Infections + Rt)

# Code
```
import jax.numpy as jnp
import pyrenew.arrayutils as au
from numpy.typing import ArrayLike
from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable

class RtInfectionsRenewalSample(NamedTuple):
    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    observed_infections: ArrayLike | None = None

class RtInfectionsRenewalModel(Model):
    def __init__(
        self,
        latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        infection_obs_process_rv: RandomVariable = None,
    ) -> None:
        if infection_obs_process_rv is None:
            infection_obs_process_rv = NullObservation()
        RtInfectionsRenewalModel.validate(
            gen_int_rv=gen_int_rv,
            I0_rv=I0_rv,
            latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=infection_obs_process_rv,
            Rt_process_rv=Rt_process_rv,
        )
        self.gen_int_rv = gen_int_rv
        self.I0_rv = I0_rv
        self.latent_infections_rv = latent_infections_rv
        self.infection_obs_process_rv = infection_obs_process_rv
        self.Rt_process_rv = Rt_process_rv

    @staticmethod
    def validate(
        gen_int_rv: any,
        I0_rv: any,
        latent_infections_rv: any,
        infection_obs_process_rv: any,
        Rt_process_rv: any,
    ) -> None:
        _assert_sample_and_rtype(gen_int_rv, skip_if_none=False)
        _assert_sample_and_rtype(I0_rv, skip_if_none=False)
        _assert_sample_and_rtype(latent_infections_rv, skip_if_none=False)
        _assert_sample_and_rtype(infection_obs_process_rv, skip_if_none=False)
        _assert_sample_and_rtype(Rt_process_rv, skip_if_none=False)
        return None

    def sample_rt(
        self,
        **kwargs,
    ) -> tuple:
        return self.Rt_process_rv.sample(**kwargs)

    def sample_gen_int(
        self,
        **kwargs,
    ) -> tuple:
        return self.gen_int_rv.sample(**kwargs)

    def sample_I0(
        self,
        **kwargs,
    ) -> tuple:
        return self.I0_rv.sample(**kwargs)

    def sample_infections_latent(
        self,
        **kwargs,
    ) -> tuple:
        return self.latent_infections_rv.sample(**kwargs)

    def sample_infection_obs_process(
        self,
        observed_infections_mean: ArrayLike,
        data_observed_infections: ArrayLike | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple:
        return self.infection_obs_process_rv.sample(
            mu=observed_infections_mean,
            obs=data_observed_infections,
            name=name,
            **kwargs,

    def sample(
        self,
        n_timepoints_to_simulate: int | None = None,
        data_observed_infections: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> RtInfectionsRenewalSample:
        """
        Sample from the Basic Renewal Model

        Parameters
        ----------
        n_timepoints_to_simulate : int, optional
            Number of timepoints to sample.
        data_observed_infections : ArrayLike | None, optional
            Observed infections. Defaults to None.
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, if any

        Notes
        -----
        Either `data_observed_infections` or `n_timepoints_to_simulate` must be specified, not both.

        Returns
        -------
        RtInfectionsRenewalSample
        """

        if (
            n_timepoints_to_simulate is None
            and data_observed_infections is None
        ):
            raise ValueError(
                "Either n_timepoints_to_simulate or data_observed_infections "
                "must be passed."
            )
        elif (
            n_timepoints_to_simulate is not None
            and data_observed_infections is not None
        ):
            raise ValueError(
                "Cannot pass both n_timepoints_to_simulate and data_observed_infections."
            )
        elif n_timepoints_to_simulate is None:
            n_timepoints = len(data_observed_infections)
        else:
            n_timepoints = n_timepoints_to_simulate
        # Sampling from Rt (possibly with a given Rt, depending on
        # the Rt_process (RandomVariable) object.)
        Rt, *_ = self.sample_rt(
            n_timepoints=n_timepoints,
            **kwargs,
        )

        # Getting the generation interval
        gen_int, *_ = self.sample_gen_int(**kwargs)

        # Sampling initial infections
        I0, *_ = self.sample_I0(**kwargs)
        I0_size = I0.size
        # Sampling from the latent process
        latent_infections, *_ = self.sample_infections_latent(
            Rt=Rt,
            gen_int=gen_int,
            I0=I0,
            **kwargs,
        )

        if data_observed_infections is None:
            (
                observed_infections,
                *_,
            ) = self.sample_infection_obs_process(
                observed_infections_mean=latent_infections,
                data_observed_infections=data_observed_infections,
                **kwargs,
            )
        else:
            data_observed_infections = au.pad_x_to_match_y(
                data_observed_infections,
                latent_infections,
                jnp.nan,
                pad_direction="start",
            )

            (
                observed_infections,
                *_,
            ) = self.sample_infection_obs_process(
                observed_infections_mean=latent_infections[
                    I0_size + padding :
                ],
                data_observed_infections=data_observed_infections[
                    I0_size + padding :
                ],
                **kwargs,
            )

        observed_infections = au.pad_x_to_match_y(
            observed_infections,
            latent_infections,
            jnp.nan,
            pad_direction="start",
        )

        Rt = au.pad_x_to_match_y(
            Rt, latent_infections, jnp.nan, pad_direction="start"
        )
        return RtInfectionsRenewalSample(
            Rt=Rt,
            latent_infections=latent_infections,
            observed_infections=observed_infections,

```
