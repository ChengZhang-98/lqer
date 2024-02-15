import logging
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)
from transformers.models.opt.modeling_opt import (
    OPTForCausalLM,
    OPTForSequenceClassification,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralForSequenceClassification,
)
from .llama_decoder import quantize_llama_model
from .opt_decoder import quantize_opt_model
from .mistral_decoder import quantize_mistral_model

logger = logging.getLogger(__name__)


def quantize_model(model, q_config, l_config) -> None:
    if isinstance(model, (LlamaForCausalLM, LlamaForSequenceClassification)):
        q_model = quantize_llama_model(model, q_config, l_config)
    elif isinstance(model, (OPTForCausalLM, OPTForSequenceClassification)):
        q_model = quantize_opt_model(model, q_config, l_config)
    elif isinstance(model, (MistralForCausalLM, MistralForSequenceClassification)):
        q_model = quantize_mistral_model(model, q_config, l_config)
    else:
        msg = f"Model {type(model).__name__} not supported for quantization"
        raise NotImplementedError(msg)

    logger.debug("Quantized model: %s", q_model)
    return q_model
