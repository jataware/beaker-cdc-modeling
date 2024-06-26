# Description
Pad the `x` array to match the length of the `y` array, specifying padding direction and fill value.

# Code
```
from typing import NamedTuple
import jax.numpy as jnp
from jax.typing import ArrayLike

def pad_to_match(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: float = 0.0,
    pad_direction: str = 'end',
    fix_y: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    x_len = x.size
    y_len = y.size
    pad_size = abs(x_len - y_len)
    pad_width = {'start': (pad_size, 0), 'end': (0, pad_size)}.get(
        pad_direction, None
    )
    if pad_width is None:
        raise ValueError(
            'pad_direction must be either 'start' or 'end'.'
            f' Got {pad_direction}.'
        )
    if x_len > y_len:
        if fix_y:
            raise ValueError(
                'Cannot fix y when x is longer than y.'
                f' x_len: {x_len}, y_len: {y_len}.'
            )
        y = jnp.pad(y, pad_width, constant_values=fill_value)
    elif y_len > x_len:
        x = jnp.pad(x, pad_width, constant_values=fill_value)

def pad_x_to_match_y(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: float = 0.0,
    pad_direction: str = "end",
) -> ArrayLike:
    """
    Pad the `x` array at the start or end to match the length of the `y` array.

    Parameters
    ----------
    x : ArrayLike
        First array.
    y : ArrayLike
        Second array.
    fill_value : float, optional
        Value to use for padding, by default 0.0.
    pad_direction : str, optional
        Direction to pad the shorter array, either "start" or "end", by default "end".

    Returns
    -------
    Array
        Padded array.
    """
    return pad_to_match(
        x, y, fill_value=fill_value, pad_direction=pad_direction, fix_y=True

```
