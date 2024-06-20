# Description
Generate infections according to a renewal process with infection feedback, adjusting the timeseries of the reproduction number R(t) using a feedback mechanism.

# Code
```
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyrenew.convolve import new_double_convolve_scanner

def compute_infections_from_rt_with_feedback(
    I0: ArrayLike,
    Rt_raw: ArrayLike,
    infection_feedback_strength: ArrayLike,
    reversed_generation_interval_pmf: ArrayLike,
    reversed_infection_feedback_pmf: ArrayLike,
) -> tuple:
    r"""
    Generate infections according to
    a renewal process with infection
    feedback (generalizing Asher 2018:
    https://doi.org/10.1016/j.epidem.2017.02.009)

    Parameters
    ----------
    I0 : ArrayLike
        Array of initial infections of the
        same length as the generation interval
        pmf vector.
    Rt_raw : ArrayLike
        Timeseries of raw R(t) values not
        adjusted by infection feedback
    infection_feedback_strength : ArrayLike
        Strength of the infection feedback.
        Either a scalar (constant feedback
        strength in time) or a vector representing
        the infection feedback strength at a
        given point in time.
    reversed_generation_interval_pmf : ArrayLike
        discrete probability mass vector
        representing the generation interval
        of the infection process, where the final
        entry represents an infection 1 time unit in the
        past, the second-to-last entry represents
        an infection two time units in the past, etc.
    reversed_infection_feedback_pmf : ArrayLike
        discrete probability mass vector
        representing the infection feedback
        process, where the final entry represents
        the relative contribution to infection
        feedback from infections that occurred
        1 time unit in the past, the second-to-last
        entry represents the contribution from infections
        that occurred 2 time units in the past, etc.

    Returns
    -------
    tuple
        A tuple `(infections, Rt_adjusted)`,
        where `Rt_adjusted` is the infection-feedback-adjusted
        timeseries of the reproduction number R(t) and
        infections is the incident infection timeseries.

    Notes
    -----
    This function implements the following renewal process:

    .. math::

        I(t) & = \mathcal{R}(t)\sum_{\tau=1}^{T_g}I(t - \tau)g(\tau)

        \mathcal{R}(t) & = \mathcal{R}^u(t)\exp\left(\gamma(t)\
            \sum_{\tau=1}^{T_f}I(t - \tau)f(\tau)\right)

    where :math:`\mathcal{R}(t)` is the reproductive number,
    :math:`\gamma(t)` is the infection feedback strength,
    :math:`T_g` is the max-length of the
    generation interval, :math:`\mathcal{R}^u(t)` is the raw reproduction
    number, :math:`f(t)` is the infection feedback pmf, and :math:`T_f`
    is the max-length of the infection feedback pmf.

    Note that negative :math:`\gamma(t)` implies
    that recent incident infections reduce :math:`\mathcal{R}(t)`
    below its raw value in the absence of feedback, while
    positive :math:`\gamma` implies that recent incident infections
    _increase_ :math:`\mathcal{R}(t)` above its raw value, and
    :math:`gamma(t)=0` implies no feedback.

    In general, negative :math:`\gamma` is the more common modeling
    choice, as it can be used to model susceptible depletion,
    reductions in contact rate due to awareness of high incidence,
    et cetera.
    """
    feedback_scanner = new_double_convolve_scanner(
        arrays_to_convolve=(
            reversed_infection_feedback_pmf,
            reversed_generation_interval_pmf,
        ),
        transforms=(ExpTransform(), IdentityTransform()),
    )
    latest, infs_and_R_adj = jax.lax.scan(
        f=feedback_scanner,
        init=I0,
        xs=(infection_feedback_strength, Rt_raw),
    )

    infections, R_adjustment = infs_and_R_adj

```
