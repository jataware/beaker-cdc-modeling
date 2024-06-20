# Description
An example of sampling from the HospitalAdmissions model to simulate or fit observed hospital admissions data.

# Code
```
import jax.numpy as jnp
import pyrenew.arrayutils as au
from jax.typing import ArrayLike
from pyrenew.metaclass import Model, RandomVariable, _assert_sample_and_rtype
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel

class HospModelSample(NamedTuple):
    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    infection_hosp_rate: float | None = None
    latent_hosp_admissions: ArrayLike | None = None
    observed_hosp_admissions: ArrayLike | None = None

class HospitalAdmissionsModel(Model):
    def __init__(
        self, latent_hosp_admissions_rv: RandomVariable, latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable, I0_rv: RandomVariable, Rt_process_rv: RandomVariable,
        hosp_admission_obs_process_rv: RandomVariable) -> None:
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int_rv=gen_int_rv, I0_rv=I0_rv, latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=None, Rt_process_rv=Rt_process_rv)
        HospitalAdmissionsModel.validate(latent_hosp_admissions_rv, hosp_admission_obs_process_rv)
        self.latent_hosp_admissions_rv = latent_hosp_admissions_rv
        self.hosp_admission_obs_process_rv = hosp_admission_obs_process_rv
    @staticmethod
    def validate(latent_hosp_admissions_rv, hosp_admission_obs_process_rv) -> None:
        _assert_sample_and_rtype(latent_hosp_admissions_rv, skip_if_none=False)
        _assert_sample_and_rtype(hosp_admission_obs_process_rv, skip_if_none=True)
        return None
    def sample_latent_hosp_admissions(self, latent_infections: ArrayLike, **kwargs) -> tuple:
        return self.latent_hosp_admissions_rv.sample(latent_infections=latent_infections, **kwargs)
    def sample_admissions_process(self, observed_hosp_admissions_mean: ArrayLike,
                                  data_observed_hosp_admissions: ArrayLike, name: str | None = None,
                                  **kwargs) -> tuple:
        return self.hosp_admission_obs_process_rv.sample(
            mu=observed_hosp_admissions_mean,
            obs=data_observed_hosp_admissions,
            name=name,

    def sample(
        self,
        n_timepoints_to_simulate: int | None = None,
        data_observed_hosp_admissions: ArrayLike | None = None,
        padding: int = 0,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the HospitalAdmissions model

        Parameters
        ----------
        n_timepoints_to_simulate : int, optional
            Number of timepoints to sample (passed to the basic renewal model).
        data_observed_hosp_admissions : ArrayLike, optional
            The observed hospitalization data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        padding : int, optional
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        HospModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        sample_latent_hosp_admissions : To sample latent hospitalization process
        sample_observed_admissions : For sampling observed hospital admissions
        """
        if (
            n_timepoints_to_simulate is None
            and data_observed_hosp_admissions is None
        ):
            raise ValueError(
                "Either n_timepoints_to_simulate or data_observed_hosp_admissions "
                "must be passed."
            )
        elif (
            n_timepoints_to_simulate is not None
            and data_observed_hosp_admissions is not None
        ):
            raise ValueError(
                "Cannot pass both n_timepoints_to_simulate and data_observed_hosp_admissions."
            )
        elif n_timepoints_to_simulate is None:
            n_timepoints = len(data_observed_hosp_admissions)
        else:
            n_timepoints = n_timepoints_to_simulate

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_timepoints_to_simulate=n_timepoints,
            data_observed_infections=None,
            padding=padding,
            **kwargs,
        )

        # Sampling the latent hospital admissions
        (
            infection_hosp_rate,
            latent_hosp_admissions,
            *_,
        ) = self.sample_latent_hosp_admissions(
            latent_infections=basic_model.latent_infections,
            **kwargs,
        )
        i0_size = len(latent_hosp_admissions) - n_timepoints
        if self.hosp_admission_obs_process_rv is None:
            observed_hosp_admissions = None
        else:
            if data_observed_hosp_admissions is None:
                (
                    observed_hosp_admissions,
                    *_,
                ) = self.sample_admissions_process(
                    observed_hosp_admissions_mean=latent_hosp_admissions,
                    data_observed_hosp_admissions=data_observed_hosp_admissions,
                    **kwargs,
                )
            else:
                data_observed_hosp_admissions = au.pad_x_to_match_y(
                    data_observed_hosp_admissions,
                    latent_hosp_admissions,
                    jnp.nan,
                    pad_direction="start",
                )

                (
                    observed_hosp_admissions,
                    *_,
                ) = self.sample_admissions_process(
                    observed_hosp_admissions_mean=latent_hosp_admissions[
                        i0_size + padding :
                    ],
                    data_observed_hosp_admissions=data_observed_hosp_admissions[
                        i0_size + padding :
                    ],
                    **kwargs,
                )

        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            infection_hosp_rate=infection_hosp_rate,
            latent_hosp_admissions=latent_hosp_admissions,
            observed_hosp_admissions=observed_hosp_admissions,

```
