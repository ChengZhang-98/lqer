import math
import re
from argparse import ArgumentParser
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

DECODER_LAYER_NAME_PATTERNS = [r"layers\.[0-9]+"]


def main():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()


    config=AutoConfig.from_pretrained(args.model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map_unbalanced = infer_auto_device_map(
        model,
        no_split_module_classes=model._no_split_modules,
    )
    decoder_layers = [k for k in device_map_unbalanced.keys() if any(re.search(p, k) for p in DECODER_LAYER_NAME_PATTERNS)]
    if len(decoder_layers) == 0:
        raise ValueError("No matched decoder layers found")
    num_devices = torch.cuda.device_count()
    num_decoder_layers_per_device = math.ceil(len(decoder_layers) / num_devices)

    device_map_balanced = {}
    decoder_layer_cnt = 0
    for k in device_map_unbalanced.keys():
        device_map_balanced[k] = decoder_layer_cnt // num_decoder_layers_per_device
        if k in decoder_layers:
            decoder_layer_cnt += 1
        # print(f"{k}, decoder_layer_cnt: {decoder_layer_cnt}, device_map_balanced[k]: {device_map_balanced[k]}")

    print(f"Model: {args.model_name}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_devices: {num_devices}")
    print(f"num_decoder_layers_per_device: {num_decoder_layers_per_device}")
    print(f"device_map_balanced: {str(device_map_balanced)}")

if __name__ == "__main__":
    main()