from .block_fp import block_fp_quantizer
from .integer import integer_quantizer
from .minifloat import minifloat_ieee_quantizer
from .passthrough import passthrough_quantizer


def get_quantizer(name: str):
    match name:
        case "passthrough":
            return passthrough_quantizer
        case "block_fp":
            return block_fp_quantizer
        case "integer":
            return integer_quantizer
        case "minifloat":
            return minifloat_ieee_quantizer
        case _:
            raise ValueError(f"quantizer {name} not supported")
