# Description
Function to plot the posterior distribution of a variable. The function takes the variable name, draws DataFrame, observed signal (optional), and various plot parameters, and returns a Matplotlib figure.

# Code
```
matplotlib.pyplot as plt
numpy as np
polars as pl

def plot_posterior(
    var: str,
    draws: pl.DataFrame,
    obs_signal: ArrayLike = None,
    ylab: str = None,
    xlab: str = "Time",
    samples: int = 50,
    figsize: list = [4, 5],
    draws_col: str = "darkblue",
    obs_col: str = "black",
) -> plt.Figure:
    """
    Plot the posterior distribution of a variable

    Parameters
    ----------
    var : str
        Name of the variable to plot
    model : Model
        Model object
    obs_signal : ArrayLike, optional
        Observed signal to plot as reference
    ylab : str, optional
        Label for the y-axis
    xlab : str, optional
        Label for the x-axis
    samples : int, optional
        Number of samples to plot
    figsize : list, optional
        Size of the figure
    draws_col : str, optional
        Color of the draws
    obs_col : str, optional
        Color of observations column.

    Returns
    -------
    plt.Figure
    """

    if ylab is None:
        ylab = var

    fig, ax = plt.subplots(figsize=figsize)

    # Reference signal (if any)
    if obs_signal is not None:
        ax.plot(obs_signal, color=obs_col)

    samp_ids = np.random.randint(size=samples, low=0, high=999)

    for samp_id in samp_ids:
        sub_samps = draws.filter(pl.col("draw") == samp_id).sort(
            pl.col("time")
        )
        ax.plot(
            sub_samps.select("time").to_numpy(),
            sub_samps.select(var).to_numpy(),
            color=draws_col,
            alpha=0.1,
        )

    # Some labels
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Adding a legend
    ax.plot([], [], color=draws_col, alpha=0.9, label="Posterior samples")

    if obs_signal is not None:
        ax.plot([], [], color=obs_col, label="Observed signal")

    ax.legend()


```
