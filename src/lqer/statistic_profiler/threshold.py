import math
import torch


class ThresholdHookFactory:
    def __init__(self, threshold: float, seq_len: int):
        self.threshold = threshold
        self.seq_len = seq_len
        self.results = {}
        self.is_profiled = {}

    def get_threshold_hook(
        self, name: str, in_features: int, out_features: int
    ) -> callable:
        """
        X @ W^T
        X: [batch_size, seq_len, in_features]
        W: [out_features, in_features]
        W^T: [in_features, out_features]
        """
        self.results[name] = {
            "weight_shape": (out_features, in_features),
            "high_precision_weight_shape": None,
            "low_precision_weight_shape": None,
            "high_precision_activation_shape": None,
            "low_precision_activation_shape": None,
            "running_num_x_cols_hp": None,
        }
        self.is_profiled[name] = False

        @torch.no_grad()
        def threshold_hook(
            module: torch.nn.Linear,
            input: torch.Tensor,
            output: torch.Tensor,
        ):
            x = input[0]
            assert x.ndim >= 2
            is_large = x.abs().ge(self.threshold)
            num_x_cols_hp = is_large.view(-1, x.shape[-1]).any(dim=0).sum().item()
            if self.results[name]["running_num_x_cols_hp"] is None:
                self.results[name]["running_num_x_cols_hp"] = [num_x_cols_hp]
            else:
                self.results[name]["running_num_x_cols_hp"].append(num_x_cols_hp)

            self.is_profiled[name] = True

        return threshold_hook

    def is_all_profiled(self) -> bool:
        return all(self.is_profiled.values())

    def get_threshold_dict(self) -> dict[str, dict]:
        assert self.is_all_profiled(), "Not all thresholds are profiled."

        for name, result in self.results.items():
            reduced_x = result.pop("running_num_x_cols_hp")
            x_n_cols_hp = math.ceil(sum(reduced_x) / len(reduced_x))

            w_shape = result["weight_shape"]

            result["high_precision_weight_shape"] = (w_shape[0], x_n_cols_hp)
            result["low_precision_weight_shape"] = (
                w_shape[0],
                w_shape[1] - x_n_cols_hp,
            )
            result["high_precision_activation_shape"] = (
                self.seq_len,
                x_n_cols_hp,
            )
            result["low_precision_activation_shape"] = (
                self.seq_len,
                w_shape[1] - x_n_cols_hp,
            )
            result["threshold"] = self.threshold
            result["seq_len"] = self.seq_len
            result["num_activation_columns_in_high_precision"] = x_n_cols_hp

        return self.results


def register_threshold_hooks(
    model: torch.nn.Module, threshold: float, seq_len: int
) -> dict:
    threshold_hook_factory = ThresholdHookFactory(threshold, seq_len=seq_len)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        name = name + ".threshold"
        module.register_forward_hook(
            threshold_hook_factory.get_threshold_hook(
                name, module.in_features, module.out_features
            )
        )
    return threshold_hook_factory
