# Contribution 

## Tests

```bash
make tests
```

## Developpers

### Code coverage

Whenever you implement a new functionality you should reach a proper code coverage. Every pull requests will be rejected if the code coverage doesn't reach *90%*.

### Docstring formatting

The doc generation tool used is portray which handle markdown format

```python
def func(a, b):
  """Describe my function

  Arguments:

  - *a*: A param a description
  - *b*: A param b description

  Returns:

  The sum of a + b

  Raises: (If exception are raised)

  Exceptions:
  1
  """
  return a + b
```

### The code formatting used is yapf

The config are automatically loaded from [.style.yapf](./.style.yapf)

YAPF tries very hard to get the formatting correct. But for some code, it won't be as good as hand-formatting. In particular, large data literals may become horribly disfigured under YAPF.

The reasons for this are manyfold. In short, YAPF is simply a tool to help with development. It will format things to coincide with the style guide, but that may not equate with readability.

What can be done to alleviate this situation is to indicate regions YAPF should ignore when reformatting something:

```python
# yapf: disable
FOO = {
    # ... some very large, complex data literal.
}

BAR = [
    # ... another large data literal.
]
# yapf: enable
```

You can also disable formatting for a single literal like this:

```python
BAZ = {
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
}  # yapf: disable
```

In addition of this, it's recommended to have an automatic formatter of the
imports. Imports of the same module should be imported together:

```python

from libs.fooo.resnet_v1_101 import (create_resnet, support_utils_resnet_foo, upload_foo)

```

Imports should be structured in 3 parts, each separated by a blank line:

```python
import os  # Base package

import keras  # External package from pip

from libs import foo  # import package inner modules
```

Finally, it is prefered that the imports are sorted alphabetically.
