import torch

SCALE_CLAMP_MIN = 1e-4


class ScaleHookFactoryBase:
    def __init__(self):
        self.scales = {}
        self.is_profiled = {}

    def get_scale_hook(self, name: str, in_features: int) -> callable:
        raise NotImplementedError("get_scale_hook is not implemented.")

    def is_all_profiled(self) -> bool:
        return all(self.is_profiled.values())

    def get_scale_dict(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("get_scale_dict is not implemented.")


class ScaleHookFactoryMeanAbs(ScaleHookFactoryBase):
    def get_scale_hook(self, name: str, in_features: int) -> callable:
        self.scales[name] = torch.zeros(in_features, dtype=torch.float32)
        self.is_profiled[name] = False

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0].float()
            scale = self.scales[name].to(x.device)
            # x.shape = [batch_size, seq_len, in_features]
            # x_abs.shape = [in_features]
            x_abs = x.abs().view(-1, x.shape[-1]).mean(0)
            # scale.shape = [in_features]
            scale = torch.maximum(scale, x_abs)
            self.scales[name] = scale
            self.is_profiled[name] = True

        return scale_hook

    def get_scale_dict(self) -> dict[str, torch.Tensor]:
        assert self.is_all_profiled(), "Not all scales are profiled."
        for name, scale in self.scales.items():
            scale = scale.clamp(min=SCALE_CLAMP_MIN)
            scale = scale / torch.sqrt(scale.min() * scale.max())
            self.scales[name] = scale

        return self.scales


def register_scale_hooks(
    model: torch.nn.Module, mode: str = "mean(abs())"
) -> ScaleHookFactoryMeanAbs:
    if mode == "mean(abs())":
        scale_hook_factory = ScaleHookFactoryMeanAbs()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        name = name + ".scale"
        module.register_forward_hook(
            scale_hook_factory.get_scale_hook(name, module.in_features)
        )
    return scale_hook_factory
