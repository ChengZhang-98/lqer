import torch


class ScaleHookFactory:
    def __init__(self):
        self.scales = {}
        self.is_profiled = {}

    def get_scale_hook(self, name: str, in_features: int) -> callable:
        self.scales[name] = torch.zeros(in_features)
        self.is_profiled[name] = False

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            scale = self.scales[name].to(x.device)
            # x.shape = [batch_size, seq_len, in_features]
            # x_abs.shape = [in_features]
            x_abs = x.abs().view(-1, x.shape[-1]).mean(0)
            # scale.shape = [in_features]
            scale = torch.maximum(scale, x_abs)
            self.scales[name] = scale
            self.is_profiled[name] = True

        return scale_hook

    def is_all_profiled(self) -> bool:
        return all(self.is_profiled.values())

    def get_scale_dict(self) -> dict[str, torch.Tensor]:
        assert self.is_all_profiled(), "Not all scales are profiled."
        return self.scales


def register_scale_hooks(model: torch.nn.Module) -> ScaleHookFactory:
    scale_hook_factory = ScaleHookFactory()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        name = name + ".scale"
        module.register_forward_hook(
            scale_hook_factory.get_scale_hook(name, module.in_features)
        )
    return scale_hook_factory
