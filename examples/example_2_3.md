# Description
Broadcast arrays periodically using either repeat or tile, considering period size and starting point.

# Code
```
from typing import NamedTuple
import jax.numpy as jnp

class PeriodicBroadcaster:
    r"""
    Broadcast arrays periodically using either repeat or tile,
    considering period size and starting point.
    """

    def __init__(
        self,
        offset: int,
        period_size: int,
        broadcast_type: str,
    ) -> None:
        """
        Default constructor for PeriodicBroadcaster class.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        broadcast_type : str
            Type of broadcasting to use, either "repeat" or "tile".

        Notes
        -----
        See the sample method for more information on the broadcasting types.

        Returns
        -------
        None
        """

        self.validate(
            offset=offset,
            period_size=period_size,
            broadcast_type=broadcast_type,
        )

        self.period_size = period_size
        self.offset = offset
        self.broadcast_type = broadcast_type

        return None

    @staticmethod
    def validate(offset: int, period_size: int, broadcast_type: str) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        offset : int
            Relative point at which data starts, must be between 0 and
            period_size - 1.
        period_size : int
            Size of the period.
        broadcast_type : str
            Type of broadcasting to use, either "repeat" or "tile".

        Returns
        -------
        None
        """

        # Period size should be a positive integer
        assert isinstance(
            period_size, int
        ), f"period_size should be an integer. It is {type(period_size)}."

        assert (
            period_size > 0
        ), f"period_size should be a positive integer. It is {period_size}."

        # Data starts should be a positive integer
        assert isinstance(
            offset, int
        ), f"offset should be an integer. It is {type(offset)}."

        assert (
            0 <= offset
        ), f"offset should be a positive integer. It is {offset}."

        assert offset <= period_size - 1, (
            "offset should be less than or equal to period_size - 1."
            f"It is {offset}. It should be less than or equal "
            f"to {period_size - 1}."
        )

        # Broadcast type should be either "repeat" or "tile"
        assert broadcast_type in ["repeat", "tile"], (
            "broadcast_type should be either 'repeat' or 'tile'. "
            f"It is {broadcast_type}."
        )

        return None

    def __call__(
        self,
        data: ArrayLike,
        n_timepoints: int,
    ) -> ArrayLike:
        """
        Broadcast the data to the given number of timepoints
        considering the period size and starting point.

        Parameters
        ----------
        data: ArrayLike
            Data to broadcast.
        n_timepoints : int
            Duration of the sequence.

        Notes
        -----
        The broadcasting is done by repeating or tiling the data. When
        self.broadcast_type = "repeat", the function will repeat each value of the data `self.period_size` times until it reaches `n_timepoints`. When self.broadcast_type = "tile", the function will tile the data until it reaches `n_timepoints`.

        Using the `offset` parameter, the function will start the broadcast from the `offset`-th element of the data. If the data is shorter than `n_timepoints`, the function will repeat or tile the data until it reaches `n_timepoints`.

        Returns
        -------
        ArrayLike
            Broadcasted array.
        """

        if self.broadcast_type == "repeat":
            return jnp.repeat(data, self.period_size)[
                self.offset : (self.offset + n_timepoints)
            ]
        elif self.broadcast_type == "tile":
            return jnp.tile(
                data, int(jnp.ceil(n_timepoints / self.period_size))

```
