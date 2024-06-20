# Description
Example demonstrating how to sample hospital admissions from a given set of latent infections using the HospitalAdmissions class.

# Code
```
import jax.numpy as jnp
import numpyro as npro
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable
from typing import Any, NamedTuple

class HospitalAdmissionsSample(NamedTuple):
    infection_hosp_rate: float | None = None
    latent_hospital_admissions: ArrayLike | None = None

    def __repr__(self):
        return f"HospitalAdmissionsSample(infection_hosp_rate={self.infection_hosp_rate}, latent_hospital_admissions={self.latent_hospital_admissions})"

class HospitalAdmissions(RandomVariable):
    def __init__(
        self,
        infection_to_admission_interval_rv: RandomVariable,
        infect_hosp_rate_rv: RandomVariable,
        latent_hospital_admissions_varname: str = "latent_hospital_admissions",
        day_of_week_effect_rv: RandomVariable | None = None,
        hosp_report_prob_rv: RandomVariable | None = None,
    ) -> None:
        if day_of_week_effect_rv is None:
            day_of_week_effect_rv = DeterministicVariable(1, "weekday_effect")
        if hosp_report_prob_rv is None:
            hosp_report_prob_rv = DeterministicVariable(1, "hosp_report_prob")

        HospitalAdmissions.validate(
            infect_hosp_rate_rv,
            day_of_week_effect_rv,
            hosp_report_prob_rv,
        )

        self.latent_hospital_admissions_varname = latent_hospital_admissions_varname
        self.infect_hosp_rate_rv = infect_hosp_rate_rv
        self.day_of_week_effect_rv = day_of_week_effect_rv
        self.hosp_report_prob_rv = hosp_report_prob_rv
        self.infection_to_admission_interval_rv = infection_to_admission_interval_rv

    @staticmethod
    def validate(
        infect_hosp_rate_rv: Any,
        day_of_week_effect_rv: Any,
        hosp_report_prob_rv: Any,
    ) -> None:
        assert isinstance(infect_hosp_rate_rv, RandomVariable)
        assert isinstance(day_of_week_effect_rv, RandomVariable)
        assert isinstance(hosp_report_prob_rv, RandomVariable)

    def sample(
        self,
        latent_infections: ArrayLike,
        **kwargs,
    ) -> HospitalAdmissionsSample:
        """
        Samples from the observation process

        Parameters
        ----------
        latent : ArrayLike
            Latent infections.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        HospitalAdmissionsSample
        """

        infection_hosp_rate, *_ = self.infect_hosp_rate_rv.sample(**kwargs)

        infection_hosp_rate_t = infection_hosp_rate * latent_infections

        (
            infection_to_admission_interval_rv,
            *_,
        ) = self.infection_to_admission_interval_rv.sample(**kwargs)

        latent_hospital_admissions = jnp.convolve(
            infection_hosp_rate_t,
            infection_to_admission_interval_rv,
            mode="full",
        )[: infection_hosp_rate_t.shape[0]]

        # Applying the day of the week effect
        latent_hospital_admissions = (
            latent_hospital_admissions
            * self.day_of_week_effect_rv.sample(**kwargs)[0]
        )

        # Applying probability of hospitalization effect
        latent_hospital_admissions = (
            latent_hospital_admissions
            * self.hosp_report_prob_rv.sample(**kwargs)[0]
        )

        npro.deterministic(
            self.latent_hospital_admissions_varname, latent_hospital_admissions
        )

        return HospitalAdmissionsSample(
            infection_hosp_rate, latent_hospital_admissions

```
