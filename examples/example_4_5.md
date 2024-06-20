# Description
Minimal example of a `RandomVariable` class using `numpyro` distribution.

# Code
```

from pyrenew.metaclass import RandomVariable


class MyNormal(RandomVariable):
    def __init__(self, loc, scale):
        self.validate(scale)
        self.loc = loc
        self.scale = scale
        return None

    @staticmethod
    def validate(self):
        if self.scale <= 0:
            raise ValueError("Scale must be positive")
        return None

    def sample(self, **kwargs):

```
