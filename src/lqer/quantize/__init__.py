"""
How's quantization config looks like?

```python

q_config = {
    "name": "flexible",
    "default": {
        "name": "block_fp",
        "width": 12,
        "exponent_width": 8,
        "exponent_bias": None,
        "block_size": [16],
    },
    "x_quantizer": {
        "name": "block_fp",
        ...
    },
    "w_quantizer": {
        "name": "block_fp",
        ...
    }
}


q_config = {
    "name": integer,
    "x_quantizer": {
        "width": 8,
        "frac_width": 4,
    },
    "w_quantizer": {
        "width": 8,
        "frac_width": 6,
    }
}

```

"""

from .quantized_functions import get_quantized_func
from .quantized_layers import get_quantized_layer_cls
from .quantizers import get_quantizer
