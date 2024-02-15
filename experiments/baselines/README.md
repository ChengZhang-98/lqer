# Setup Env for Baselines

- Please following HuggingFace Quantization Guide to create an environment for quantization baselines. AutoGPTQ may not work on newest pytorch/transformers.
- Installing AutoAWQ may automatically install lm-eval-harness, which conflicts with the lm-eval-harness in `software/submodules/submodules/lm-evaluation-harness`. If so, you can remove the `lm-eval-harness` installed by AutoAWQ by running `pip uninstall lm-eval`.

## Checkpoints

### Checkpoints for HuggingFace Quantization Plugins

| Model | HF qaunt_method | Repo |
| :--- | :---: | ---: |
| llama-7b | - / LLM.intX() | huggyllama/llama-7b |
| llama-7b| AWQ | TheBloke/LLaMA-7b-AWQ |
| llama-7b | GPTQ | TheBloke/LLaMa-7B-GPTQ |
| llama-13b | - / LLM.intX() | huggyllama/llama-13b |
| llama-13b | AWQ | TheBloke/LLaMa-13B-AWQ |
| llama-13b | GPTQ | TheBloke/LLaMa-13B-GPTQ |
| llama-2-7b | - / LLM.intX() | meta-llama/Llama-2-7b-hf|
| llama-2-7b | AWQ | TheBloke/Llama-2-7B-AWQ |
| llama-2-7b | GPTQ | TheBloke/Llama-2-7B-GPTQ |
| llama-2-13b | - / LLM.intX() | meta-llama/Llama-2-13b-hf |
| llama-2-13b | AWQ | TheBloke/Llama-2-13B-AWQ |
| llama-2-13b | GPTQ | TheBloke/Llama-2-13B-GPTQ |
| vicuna-7b-v1.5 | - / LLM.intX() | lmsys/vicuna-7b-v1.5 |
| vicuna-7b-v1.5 | AWQ | TheBloke/Vicuna-7B-v1.5-AWQ |
| vicuna-7b-v1.5 | GPTQ | TheBloke/vicuna-7B-v1.5-GPTQ |
| vicuna-13b-v1.5 | - / LLM.intX() | lmsys/vicuna-13b-v1.5 |
| vicuna-13b-v1.5 | AWQ | TheBloke/vicuna-13B-v1.5-AWQ|
| vicuna-13b-v1.5 | GPTQ | TheBloke/vicuna-13B-v1.5-GPTQ |

### Checkpoints for AutoGPTQ

| Model | Repo |
| :--- | ---: |
| OPT-6.7b | iproskurina/opt-6.7b-gptq-4bit |
| OPT-13b | iproskurina/opt-13b-gptq-4bit |