import torch


def collect_linear_weights(
    state_dict: dict[str, torch.nn.Parameter], model: torch.nn.Module
):
    named_models = dict(model.named_modules())

    filtered_state_dict = {}

    for name, weight in state_dict.items():
        layer_name = ".".join(name.split(".")[:-1])
        if isinstance(named_models[layer_name], torch.nn.Linear):
            filtered_state_dict[name] = weight

    return filtered_state_dict
