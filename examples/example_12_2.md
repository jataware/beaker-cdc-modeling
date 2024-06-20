# Description
Apply the logistic susceptibility adjustment to a potential new incidence I_unadjusted as proposed in Bhatt et al 2023.

# Code
```

def logistic_susceptibility_adjustment(
    I_raw_t: float,
    frac_susceptible: float,
    n_population: float,
) -> float:
    """
    Apply the logistic susceptibility
    adjustment to a potential new
    incidence I_unadjusted proposed in
    equation 6 of Bhatt et al 2023 [1]_

    Parameters
    ----------
    I_raw_t : float
        The "unadjusted" incidence at time t,
        i.e. the incidence given an infinite
        number of available susceptible individuals.
    frac_susceptible : float
        fraction of remaining susceptible individuals
        in the population
    n_population : float
        Total size of the population.

    Returns
    -------
    float
        The adjusted value of I(t)

    References
    ----------
    .. [1] Bhatt, Samir, et al.
    "Semi-mechanistic Bayesian modelling of
    COVID-19 with renewal processes."
    Journal of the Royal Statistical Society
    Series A: Statistics in Society 186.4 (2023): 601-615.
    https://doi.org/10.1093/jrsssa/qnad030
    """
    approx_frac_infected = 1 - jnp.exp(-I_raw_t / n_population)

```
