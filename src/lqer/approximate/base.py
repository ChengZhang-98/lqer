import torch
import torch.nn as nn


class WeightApproximatorBase(nn.Module):
    """
    Y = X * W.T <= X * (W_q.T + AB)

    W.T <= W_q.T + AB
    Q_error = W.T - W_q.T <= AB

    ----------------------------
    Attributes:
    - name: str
    - W: nn.Parameter
    - A: nn.Parameter
    - B: nn.Parameter
    - rank: int
    - W_quantizer: callable
    - A_quantizer: callable
    - B_quantizer: callable
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
        super().__init__()
        self.name = name
        self.W = nn.Parameter(weight, requires_grad=False)
        self.A = nn.Parameter(torch.zeros(weight.shape[1], rank))
        self.B = nn.Parameter(torch.randn(rank, weight.shape[0]))
        self.rank = rank
        self.W_quantizer = W_quantizer
        self.A_quantizer = A_quantizer
        self.B_quantizer = B_quantizer

    @torch.no_grad()
    def q_error_T(self) -> torch.Tensor:
        """
        ideal quantization error = (W - W_q)^T
        we use A @ B to approximate it
        """
        return (self.W - self.W_quantizer(self.W)).transpose(0, 1)

    def approximate(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def approximated_q_error_T(self) -> torch.Tensor:
        raise NotImplementedError

    def approximated_W(self) -> torch.Tensor:
        raise NotImplementedError


class ModelApproximatorBase:
    requires_scale_dict: bool = False

    def __init__(self, state_dict: dict[str, torch.Tensor], config: dict) -> None:
        self.config = config
        self.approximators: dict[str, WeightApproximatorBase] = {}
        self._post_init_setup(state_dict, config)

    def __len__(self):
        return len(self.approximators)

    def _post_init_setup(self, state_dict: dict[str, torch.Tensor], config: dict):
        """
        Create approximators
        """
        raise NotImplementedError

    def compute(self, delete_after_compute: bool = True) -> dict[str, torch.Tensor]:
        """
        Update state_dict of each approximator
        """
        raise NotImplementedError

    def to(self, target: torch.device | torch.dtype) -> None:
        for approximator in self.approximators.values():
            approximator.to(target)