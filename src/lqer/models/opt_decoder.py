import logging
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.opt.modeling_opt import (
    ACT2FN,
    OPTConfig,
    OPTForCausalLM,
    OPTForSequenceClassification,
    OPTForQuestionAnswering,
)
from ..quantize import get_quantized_func, get_quantized_layer_cls

logger = logging.getLogger(__name__)


class OPTQuantizedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        q_config=None,
        l_config=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # fmt: off
        self.k_proj = get_quantized_layer_cls("linear", q_config["k_proj"])(embed_dim, embed_dim, bias=bias, q_config=q_config["k_proj"], l_config=l_config["k_proj"] if l_config is not None else None)
        self.v_proj = get_quantized_layer_cls("linear", q_config["v_proj"])(embed_dim, embed_dim, bias=bias, q_config=q_config["v_proj"], l_config=l_config["v_proj"] if l_config is not None else None)
        self.q_proj = get_quantized_layer_cls("linear", q_config["q_proj"])(embed_dim, embed_dim, bias=bias, q_config=q_config["q_proj"], l_config=l_config["q_proj"] if l_config is not None else None)
        self.out_proj = get_quantized_layer_cls("linear", q_config["out_proj"])(embed_dim, embed_dim, bias=bias, q_config=q_config["out_proj"], l_config=l_config["out_proj"] if l_config is not None else None)
        self.q_config = q_config
        self.l_config = l_config
        # fmt: on

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = get_quantized_func("bmm", self.q_config["bmm_0"])(
            query_states, key_states.transpose(1, 2), q_config=self.q_config["bmm_0"]
        )

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(
                    torch.finfo(attn_weights.dtype).min, device=attn_weights.device
                ),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # attn_output = torch.bmm(attn_probs, value_states)
        attn_output = get_quantized_func("bmm", self.q_config["bmm_1"])(
            attn_probs, value_states, q_config=self.q_config["bmm_1"]
        )

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTQuantizedDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, q_config: dict, l_config: dict):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTQuantizedAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            q_config=q_config["self_attn"],
            l_config=l_config["self_attn"] if l_config is not None else None,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        # self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        # self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        # fmt: off
        self.fc1 = get_quantized_layer_cls("linear", q_config["fc1"])(self.embed_dim, config.ffn_dim, bias=config.enable_bias, q_config=q_config["fc1"], l_config=l_config["fc1"] if l_config is not None else None)
        self.fc2 = get_quantized_layer_cls("linear", q_config["fc2"])(config.ffn_dim, self.embed_dim, bias=config.enable_bias, q_config=q_config["fc2"], l_config=l_config["fc2"] if l_config is not None else None)
        # fmt: on
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def _layer_q_config_builder(num_hidden_layers: int, q_config: dict) -> dict:
    model_layer_cfg = q_config.get("model_layer_cfg", None)
    if model_layer_cfg is None:
        assert "bmm" in q_config and "linear" in q_config
        matmul_q_cfg = q_config["bmm"]
        linear_q_cfg = q_config["linear"]
        model_layer_cfg = {
            "self_attn": {
                "k_proj": linear_q_cfg,
                "v_proj": linear_q_cfg,
                "q_proj": linear_q_cfg,
                "out_proj": linear_q_cfg,
                "bmm_0": matmul_q_cfg,
                "bmm_1": matmul_q_cfg,
            },
            "fc1": linear_q_cfg,
            "fc2": linear_q_cfg,
        }

    model_q_cfg = {}

    for layer_id in range(num_hidden_layers):
        layer_entry = f"model_layer_{layer_id}"
        if layer_entry in q_config:
            model_q_cfg[layer_entry] = deepcopy(q_config[layer_entry])
        else:
            model_q_cfg[layer_entry] = deepcopy(model_layer_cfg)
    return model_q_cfg


def _layer_l_config_builder(num_hidden_layers: int, l_config: dict) -> dict:
    model_layer_cfg = l_config.get("model_layer", None)
    if model_layer_cfg is None:
        assert "linear" in l_config
        linear_cfg = l_config["linear"]
        model_layer_cfg = {
            "self_attn": {
                "k_proj": linear_cfg,
                "v_proj": linear_cfg,
                "q_proj": linear_cfg,
                "out_proj": linear_cfg,
            },
            "fc1": linear_cfg,
            "fc2": linear_cfg,
        }

    model_l_cfg = {}
    for layer_id in range(num_hidden_layers):
        layer_entry = f"model_layer_{layer_id}"
        if layer_entry in l_config:
            model_l_cfg[layer_entry] = deepcopy(l_config[layer_entry])
        else:
            model_l_cfg[layer_entry] = deepcopy(model_layer_cfg)

    return model_l_cfg


def quantize_opt_model(
    model: OPTForCausalLM | OPTForSequenceClassification | OPTForQuestionAnswering,
    q_config: dict,
    l_config: dict,
):
    q_config = _layer_q_config_builder(model.config.num_hidden_layers, q_config)
    if l_config is not None:
        l_config = _layer_l_config_builder(model.config.num_hidden_layers, l_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.decoder.layers):
        layer_entry = f"model_layer_{layer_id}"
        q_config_layer = q_config[layer_entry]
        if l_config is None:
            l_config_layer = None
        else:
            l_config_layer = l_config[layer_entry]

        new_decoder_layer = OPTQuantizedDecoderLayer(
            model.config, q_config_layer, l_config_layer
        )
        new_decoder_layer.to(next(iter(model.parameters())).dtype)
        new_decoder_layer.load_state_dict(ori_decoder_layer.state_dict(), strict=False)
        model.model.decoder.layers[layer_id] = new_decoder_layer

    model._no_split_modules = ["OPTDecoderLayer", "OPTQuantizedDecoderLayer"]

    return model
