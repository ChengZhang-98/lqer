from tqdm import tqdm
import gc
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

SCALE_CLAMP_MIN = 1e-4


class WeightApproximatorLqerAct(WeightApproximatorBase):
    """
    We follow LLM-AWQ paper to assign salience to each row of quantization error

    ```
    diag(scale) @ (W - W_q)^T = U * S * V_T
    ```

    where diag(scale) scales each row of quantization error

    scale is a vector of size (in_features,)

    if we assign
    ```
    A = diag(scale)^-1 @ U_k[:, :r] # r is the rank
    B = S[:r,:r] @ V_T_k[:r,:]
    ```

    then

    ```
    X @ (W_q.T + AB)  = X @ (W_q.T + diag(scale)^-1 @ U_k[:, :r] @ S[:r,:r] @ V_T_k[:r,:])
                     <= X @ (W_q.T + diag(scale)^-1 @ diag(scale) @ (W - W_q)^T)
                      = X @ (W_q.T + (W - W_q)^T)
                      = X @ W.T
    ```

    ---
    """

    def __init__(
        self,
        name: str,
        weight: torch.Tensor,
        rank: int,
        W_quantizer: callable,
        A_quantizer: callable,
        B_quantizer: callable,
    ) -> None:
        super().__init__(name, weight, rank, W_quantizer, A_quantizer, B_quantizer)
        self.scale = nn.Parameter(torch.ones(self.W.shape[1], dtype=torch.float32))

    @torch.no_grad()
    def initialize_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """
        scale = [s_1, s_2, ..., s_n], n = in_features
        scale_normalized = scale / sqrt(min(scale) * max(scale))
        """
        # scale = scale.clamp(min=SCALE_CLAMP_MIN)
        # scale = scale / torch.sqrt(scale.min() * scale.max())
        self.scale.copy_(scale)
        pass

    @torch.no_grad()
    def q_error_T(self) -> torch.Tensor:
        """
        scaled quantization error = diag(scale_n) @ (W - W_q)^T,
        where diag(scale_n) scales each row of quantization error

        we use A @ B to approximate this scaled quantization error
        """

        return torch.diag(self.scale) @ super().q_error_T()

    @torch.no_grad()
    def approximate(self, *args, **kwargs) -> None:
        scaled_q_error_T = self.q_error_T()

        U, S, V_T = torch.linalg.svd(scaled_q_error_T)

        U_k = U[:, : self.rank]
        S_k = S[: self.rank]
        V_T_k = V_T[: self.rank, :]

        A = torch.diag(self.scale).inverse() @ U_k
        B = torch.diag(S_k) @ V_T_k

        self.A.data.copy_(self.A_quantizer(A))
        self.B.data.copy_(self.B_quantizer(B))

    @torch.no_grad()
    def approximated_q_error_T(self) -> torch.Tensor:
        """
        approximated quantization error = quantize(A) * quantize(B)
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


class ModelApproximatorLqerAct(ModelApproximatorBase):
    requires_scale_dict: bool = True

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

            self.approximators[w_name] = WeightApproximatorLqerAct(
                w_name,
                weight=w,
                rank=approx_config["rank"],
                W_quantizer=w_quantizer,
                A_quantizer=a_quantizer,
                B_quantizer=b_quantizer,
            )
        if len(self.approximators) == 0:
            logger.error(
                "No matched weight found. Please check the config file and the weight names."
            )

    def load_scale_dict(self, scale_dict: dict[str, torch.Tensor]):
        for w_name, approximator in self.approximators.items():
            approximator: WeightApproximatorLqerAct
            scale_name = ".".join(w_name.split(".")[:-1] + ["scale"])
            w_scale = scale_dict[scale_name]
            assert w_scale.shape == approximator.scale.shape
            approximator.initialize_scale(w_scale)

    @torch.no_grad()
    def compute(self, delete_after_compute: bool = True) -> dict[str, torch.Tensor]:
        device = self.config.get("device", "cuda:0")

        df = pd.DataFrame(
            columns=["name", "rank", "l1_norm(AB-Q_error_T)/n", "w_dim0", "w_dim1"]
        )
        error_T_dict = {}
        low_rank_dict = {}

        prog_bar = tqdm(list(self.approximators.keys()), desc="LQER-act approximation")

        for w_name in prog_bar:
            if delete_after_compute:
                approx: WeightApproximatorLqerAct = self.approximators.pop(w_name)
            else:
                approx: WeightApproximatorLqerAct = self.approximators[w_name]
            approx.to(device)
            approx.approximate()

            q_error_T = approx.q_error_T()
            l1_norm_error = torch.linalg.vector_norm(
                approx.approximated_q_error_T() - q_error_T, ord=1
            )
            l1_norm_error /= q_error_T.numel()
            l1_norm_error = l1_norm_error.cpu().item()

            approx.to("cpu")

            df.loc[len(df)] = [
                w_name,
                approx.rank,
                l1_norm_error,
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
                f"{w_name:<60}, 1/n * ||AB - Q_error^T ||_1 ={l1_norm_error:.6f}"
            )

        return {
            "df": df,
            "error_T_dict": error_T_dict,
            "low_rank_dict": low_rank_dict,
        }
