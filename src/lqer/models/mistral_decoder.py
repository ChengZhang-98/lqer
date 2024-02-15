import logging
import inspect
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.mistral.modeling_mistral import (
    ACT2FN,
    MistralConfig,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralRotaryEmbedding,
    MistralRMSNorm,
    _get_unpad_data,
    apply_rotary_pos_emb,
    repeat_kv,
)


from ..quantize import get_quantized_func, get_quantized_layer_cls

logger = logging.getLogger(__name__)

def is_flash_attn_available():
    # override the flash_attn_available function to always return False
    return False


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


class MistralQuantizedMLP(nn.Module):
    def __init__(self, config, q_config, l_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # fmt: off
        self.gate_proj = get_quantized_layer_cls("linear", q_config=q_config["gate_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=q_config["gate_proj"], l_config=l_config["gate_proj"] if l_config is not None else None)
        self.up_proj = get_quantized_layer_cls("linear", q_config=q_config["up_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=q_config["up_proj"], l_config=l_config["up_proj"] if l_config is not None else None)
        self.down_proj = get_quantized_layer_cls("linear", q_config=q_config["down_proj"])(self.intermediate_size, self.hidden_size, bias=False, q_config=q_config["down_proj"], l_config=l_config["down_proj"] if l_config is not None else None)
        # fmt: on
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralQuantizedAttention(nn.Module):
    def __init__(self, config: MistralConfig, q_config: dict, l_config: dict) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # fmt: off
        self.q_proj = get_quantized_layer_cls("linear", q_config=q_config["q_proj"])(self.hidden_size, self.num_heads * self.head_dim, bias=False, q_config=q_config["q_proj"], l_config=l_config["q_proj"] if l_config is not None else None)
        self.k_proj = get_quantized_layer_cls("linear", q_config=q_config["k_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, q_config=q_config["k_proj"], l_config=l_config["k_proj"] if l_config is not None else None)
        self.v_proj = get_quantized_layer_cls("linear", q_config=q_config["v_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, q_config=q_config["v_proj"], l_config=l_config["v_proj"] if l_config is not None else None)
        self.o_proj = get_quantized_layer_cls("linear", q_config=q_config["o_proj"])(self.num_heads * self.head_dim, self.hidden_size, bias=False, q_config=q_config["o_proj"], l_config=l_config["o_proj"] if l_config is not None else None)
        # fmt: on
        self.q_config = q_config
        self.l_config = l_config

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(
        #     query_states, key_states.transpose(2, 3)
        # ) / math.sqrt(self.head_dim)
        query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = get_quantized_func("matmul", q_config=self.q_config["matmul_0"])(
            query_states, key_states.transpose(1, 2), q_config=self.q_config["matmul_0"]
        ) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_weights = attn_weights.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        value_states = value_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_output = get_quantized_func("matmul", q_config=self.q_config["matmul_1"])(
            attn_weights, value_states, q_config=self.q_config["matmul_1"]
        )
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralQuantizedFlashAttention2(MistralQuantizedAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and hasattr(self.config, "sliding_window") is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            if hasattr(self.config, "sliding_window") and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window

                past_key = past_key_value[0]
                past_value = past_key_value[1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key much have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                past_key_value = (past_key, past_value)

                if padding_mask is not None:
                    padding_mask = padding_mask[:, slicing_tokens:]
                    padding_mask = torch.cat([padding_mask, torch.ones_like(padding_mask[:, -1:])], dim=-1)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: Mistral does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, padding_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != padding_mask.shape[-1]:
            padding_mask_num_tokens = padding_mask.shape[-1]
            padding_mask = padding_mask[:, padding_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MistralQuantizedDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, q_config: dict, l_config: dict) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = (
            MistralQuantizedAttention(
                config=config,
                q_config=q_config["self_attn"],
                l_config=l_config["self_attn"] if l_config is not None else None,
            )
            if not getattr(config, "_flash_attn_2_enabled", False)
            else MistralQuantizedFlashAttention2(
                config=config,
                q_config=q_config["self_attn"],
                l_config=l_config["self_attn"] if l_config is not None else None,
            )
        )
        self.mlp = MistralQuantizedMLP(
            config=config,
            q_config=q_config["mlp"],
            l_config=l_config["mlp"] if l_config is not None else None,
        )
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def _layer_q_config_builder(num_hidden_layers: int, q_config: dict):
    model_layer_cfg = q_config.get("model_layer", None)
    if model_layer_cfg is None:
        assert "matmul" in q_config and "linear" in q_config
        matmul_cfg = q_config["matmul"]
        linear_cfg = q_config["linear"]
        model_layer_cfg = {
            "self_attn": {
                "k_proj": linear_cfg,
                "q_proj": linear_cfg,
                "v_proj": linear_cfg,
                "o_proj": linear_cfg,
                "matmul_0": matmul_cfg,
                "matmul_1": matmul_cfg,
            },
            "mlp": {
                "gate_proj": linear_cfg,
                "up_proj": linear_cfg,
                "down_proj": linear_cfg,
            },
        }

    model_q_cfg = {}
    for layer_id in range(num_hidden_layers):
        layer_entry = f"model_layer_{layer_id}"
        if layer_entry in q_config:
            model_q_cfg[layer_entry] = deepcopy(q_config[layer_entry])
        else:
            model_q_cfg[layer_entry] = deepcopy(model_layer_cfg)
    return model_q_cfg


def _layer_l_config_builder(num_hidden_layers: int, l_config: dict):
    model_layer_cfg = l_config.get("model_layer", None)
    if model_layer_cfg is None:
        assert "linear" in l_config
        linear_cfg = l_config["linear"]
        model_layer_cfg = {
            "self_attn": {
                "k_proj": linear_cfg,
                "q_proj": linear_cfg,
                "v_proj": linear_cfg,
                "o_proj": linear_cfg,
            },
            "mlp": {
                "gate_proj": linear_cfg,
                "up_proj": linear_cfg,
                "down_proj": linear_cfg,
            },
        }

    model_l_cfg = {}
    for layer_id in range(num_hidden_layers):
        layer_entry = f"model_layer_{layer_id}"
        if layer_entry in l_config:
            model_l_cfg[layer_entry] = deepcopy(l_config[layer_entry])
        else:
            model_l_cfg[layer_entry] = deepcopy(model_layer_cfg)

    return model_l_cfg


def quantize_mistral_model(
    model: MistralForCausalLM | MistralForSequenceClassification,
    q_config: dict,
    l_config: dict,
):
    q_config = _layer_q_config_builder(model.config.num_hidden_layers, q_config)
    if l_config is not None:
        l_config = _layer_l_config_builder(model.config.num_hidden_layers, l_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.layers):
        layer_entry = f"model_layer_{layer_id}"
        layer_q_config = q_config[layer_entry]
        if l_config is None:
            layer_l_config = None
        else:
            layer_l_config = l_config[layer_entry]

        # replace the decoder layer with quantized decoder layer
        new_decoder_layer = MistralQuantizedDecoderLayer(
            model.config, layer_q_config, layer_l_config
        )
        new_decoder_layer.to(next(iter(model.parameters())).dtype)
        new_decoder_layer.load_state_dict(ori_decoder_layer.state_dict(), strict=False)
        model.model.layers[layer_id] = new_decoder_layer

    model._no_split_modules = ["MistralDecoderLayer", "MistralQuantizedDecoderLayer"]

    return model
