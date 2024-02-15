import gc
import tqdm
import logging
from copy import deepcopy
from functools import partial
import pandas as pd
import torch
import torch.nn as nn

from .base import WeightApproximatorBase, ModelApproximatorBase
from ..utils import find_matched_pattern
from ..quantize import get_quantizer

logger = logging.getLogger(__name__)


class WeightApproximatorLqerSvd(WeightApproximatorBase):
    """
    SVD-based low-rank approximation: https://web.stanford.edu/class/cs168/l/l9.pdf

    ```
    Y = X * W_q.T + X * E
      = X * W_q.T + X * (U * S * V.T)

    W_q.shape = (out_features, in_features)
    W_q.T.shape = (in_features, out_features)

    U.shape = (in_features, in_features)
    S.shape = (in_features, out_features)
    V.T.shape = (out_features, out_features)

    where q_error.T = (W - W_q).T = U * S * V.T
    ```
    """

    @torch.no_grad()
    def approximate(self, *args, **kwargs) -> None:
        q_error_T = self.q_error_T()

        U, S, V_T = torch.linalg.svd(q_error_T)

        U_k = U[:, : self.rank]
        S_k = S[: self.rank]
        V_T_k = V_T[: self.rank, :]

        self.A.copy_(self.A_quantizer(U_k))
        self.B.copy_(self.B_quantizer(torch.diag(S_k) @ V_T_k))

    @torch.no_grad()
    def approximated_q_error_T(self) -> torch.Tensor:
        """
        approximated quantization error = quantize(U_k) * quantize(S_k * V_T_k)
        """
        return torch.matmul(self.A, self.B)

    @torch.no_grad()
    def approximated_W(self) -> torch.Tensor:
        """
        approximated W = W_q + approximated_q_error_T
        """
        W_q = self.W_quantizer(self.W)
        W_q = W_q + self.approximated_q_error_T().transpose(0, 1)
        return W_q


class ModelApproximatorLqerSvd(ModelApproximatorBase):
    def _post_init_setup(self, state_dict: dict[str, torch.Tensor], config: dict):
        for w_name, w in state_dict.items():
            entry = find_matched_pattern(w_name, config["approximator"].keys())
            if entry is None:
                continue
            cfg_or_real_entry = config["approximator"][entry]

            if isinstance(cfg_or_real_entry, str):
                # use for 'default'
                approx_config = deepcopy(config["approximator"][cfg_or_real_entry])
                assert isinstance(approx_config, dict)
            else:
                assert isinstance(cfg_or_real_entry, dict)
                approx_config = deepcopy(cfg_or_real_entry)

            # fmt: off
            w_quantizer = partial(get_quantizer(approx_config["W_quantizer"].pop("name")), **approx_config["W_quantizer"])
            a_quantizer = partial(get_quantizer(approx_config["A_quantizer"].pop("name")), **approx_config["A_quantizer"])
            b_quantizer = partial(get_quantizer(approx_config["B_quantizer"].pop("name")), **approx_config["B_quantizer"])
            # fmt: on

            self.approximators[w_name] = WeightApproximatorLqerSvd(
                w_name,
                weight=w,
                rank=approx_config["rank"],
                W_quantizer=w_quantizer,
                A_quantizer=a_quantizer,
                B_quantizer=b_quantizer,
            )
        if len(self.approximators) == 0:
            logger.error(
                "No matched weight found in model approximator. Please check the config file and the weight names."
            )

    @torch.no_grad()
    def compute(self, delete_after_compute: bool = True) -> dict[str, torch.Tensor]:
        device = self.config.get("device", "cuda:0")

        df = pd.DataFrame(columns=["name", "rank", "l1_norm(AB-Q_error_T)/n", "w_dim0", "w_dim1"])
        error_T_dict = {}
        low_rank_dict = {}

        prog_bar = tqdm.tqdm(list(self.approximators.keys()), desc="LQER-svd approximation")

        for w_name in prog_bar:
            if delete_after_compute:
                approx: WeightApproximatorLqerSvd = self.approximators.pop(w_name)
            else:
                approx: WeightApproximatorLqerSvd = self.approximators[w_name]
            approx.to(device)
            approx.approximate()

            q_error_T = approx.q_error_T()
            l1_norm_error = torch.linalg.vector_norm(
                approx.approximated_q_error_T() - q_error_T, ord=1
            )
            l1_norm_error /= q_error_T.numel()

            approx.to("cpu")

            df.loc[len(df)] = [
                w_name,
                approx.rank,
                l1_norm_error.item(),
                approx.W.shape[0],
                approx.W.shape[1],
            ]
            error_T_dict[w_name] = q_error_T.cpu()
            low_rank_dict[".".join(w_name.split(".")[:-1] + ["A"])] = approx.A
            low_rank_dict[".".join(w_name.split(".")[:-1] + ["B"])] = approx.B

            if delete_after_compute:
                del approx
                gc.collect()

            prog_bar.set_description_str(
                f"{w_name:<60}, 1/n * ||AB - Q_error^T ||_1 ={l1_norm_error.item():.6f}"
            )

        return {
            "df": df,
            "error_T_dict": error_T_dict,
            "low_rank_dict": low_rank_dict,
        }
