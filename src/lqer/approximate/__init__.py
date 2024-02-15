from .lqer_svd import ModelApproximatorLqerSvd
from .lqer_act import ModelApproximatorLqerAct

# from .sgd_svd import ModelApproximatorLqerSgd

from .base import ModelApproximatorBase


def get_model_approximator_cls(name: str) -> ModelApproximatorBase:
    match name.lower():
        case "lqer-svd" | "lqer_svd":
            return ModelApproximatorLqerSvd
        case "lqer-act" | "lqer_act":
            return ModelApproximatorLqerAct
        # case "lqer-sgd" | "lqer_sgd":
        #     return ModelApproximatorLqerSgd
        case _:
            raise NotImplementedError
