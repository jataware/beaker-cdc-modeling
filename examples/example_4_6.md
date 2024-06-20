# Description
Create a named tuple for the `InfectionsWithFeedback` class.

# Code
```

# | label: data-class
from collections import namedtuple

# Creating a tuple to store the output
InfFeedbackSample = namedtuple(
    typename="InfFeedbackSample",
    field_names=["infections", "rt"],
    defaults=(None, None),

```
