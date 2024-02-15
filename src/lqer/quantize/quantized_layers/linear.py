from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import block_fp_quantizer, get_quantizer, integer_quantizer


class _LinearBase(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        q_config: dict = None,
        l_config: dict = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.q_config = q_config
        self.l_config = l_config
        self.is_ptq = q_config.get("is_ptq", False)
        self.w_is_quantized = False if self.is_ptq else None
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None

        self.A_out_quantizer = None
        self.B_out_quantizer = None

        self._setup_quantizers(q_config)
        self._setup_lqer(l_config)

    def _setup_quantizers(self, config: dict):
        """
        Setup quantizers for input, weight and bias
        """
        raise NotImplementedError

    def _setup_lqer(self, config: dict):
        """
        Setup quantizers for input, weight and bias
        """
        raise NotImplementedError

    def forward(self, x):
        if self.is_ptq:
            with torch.no_grad():
                x = self.x_quantizer(x)
                if self.w_is_quantized is False:
                    self.weight.copy_(self.w_quantizer(self.weight.data))
                    if self.bias is not None:
                        self.bias.copy_(self.b_quantizer(self.bias.data))
                    self.w_is_quantized = True
            return F.linear(x, self.weight, self.bias)
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None
            return F.linear(x, w, bias)

    @staticmethod
    def _get_quantizer_name(quantizer):
        if quantizer is None:
            return "None"
        elif isinstance(quantizer, partial):
            return quantizer.func.__name__
        else:
            return quantizer.__name__

    def __repr__(self):
        txt = "{}(in_features={}, out_features={}, bias={}, is_ptq={}, x_quantizer={}, w_quantizer={}".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.is_ptq,
            self._get_quantizer_name(self.x_quantizer),
            self._get_quantizer_name(self.w_quantizer),
        )
        return txt


class LinearFlexible(_LinearBase):
    def _setup_quantizers(self, q_config: dict):
        x_quantizer_config = deepcopy(q_config.get("x_quantizer", q_config["default"]))
        w_quantizer_config = deepcopy(q_config.get("w_quantizer", q_config["default"]))

        self.x_quantizer = partial(
            get_quantizer(x_quantizer_config.pop("name")), **x_quantizer_config
        )
        self.w_quantizer = partial(
            get_quantizer(w_quantizer_config.pop("name")), **w_quantizer_config
        )

        if self.bias is not None:
            b_quantizer_config = deepcopy(
                q_config.get("b_quantizer", q_config["default"])
            )
            self.b_quantizer = partial(
                get_quantizer(b_quantizer_config.pop("name")), **b_quantizer_config
            )

    def _setup_lqer(self, config: dict):
        pass


class LinearFlexibleLqer(LinearFlexible):
    def _setup_quantizers(self, q_config: dict):
        LinearFlexible._setup_quantizers(self, q_config)
        B_out_quantizer_config = deepcopy(
            q_config.get(
                "B_out_quantizer", q_config.get("x_quantizer", q_config["default"])
            )
        )
        A_out_quantizer_config = deepcopy(
            q_config.get(
                "A_out_quantizer", q_config.get("x_quantizer", q_config["default"])
            )
        )
        self.B_out_quantizer = partial(
            get_quantizer(B_out_quantizer_config.pop("name")),
            **B_out_quantizer_config,
        )
        self.A_out_quantizer = partial(
            get_quantizer(A_out_quantizer_config.pop("name")),
            **A_out_quantizer_config,
        )

    def _setup_lqer(self, l_config: dict):
        # Y = X_q * W_q.T + (X_q * A_q) * B_q
        #   = X_q * (W_q.T + A_q * B_q)
        # X_q.shape = (bs, seq_len, in_features)
        # W_q.shape = (out_features, in_features)
        # W_q.T.shape = (in_features, out_features)
        # B_q.shape = (rank, out_features)
        # A_q.shape = (in_features, rank)
        self.A = torch.nn.Parameter(torch.zeros(self.weight.shape[1], l_config["rank"]))
        self.B = torch.nn.Parameter(torch.zeros(l_config["rank"], self.weight.shape[0]))

    def forward(self, x):
        if self.is_ptq:
            with torch.no_grad():
                x = self.x_quantizer(x)
                if self.w_is_quantized is False:
                    self.weight.copy_(self.w_quantizer(self.weight.data))
                    if self.bias is not None:
                        self.bias.copy_(self.b_quantizer(self.bias.data))
                    self.w_is_quantized = True
                xA = self.A_out_quantizer(torch.matmul(x, self.A))
                xAB = self.B_out_quantizer(torch.matmul(xA, self.B))
                out = F.linear(x, self.weight, self.bias) + xAB
            return out
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None

            xA = self.A_out_quantizer(torch.matmul(x, self.A))
            xAB = self.B_out_quantizer(torch.matmul(xA, self.B))
            out = F.linear(x, w, bias) + xAB
            return out
