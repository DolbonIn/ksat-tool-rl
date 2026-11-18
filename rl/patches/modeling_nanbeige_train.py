# coding=utf-8
"""Training-side implementation of Nanbeige4.2 (looped transformer) for RL.

Why this file exists
--------------------
The official remote code (modeling_nanbeige.py, written for transformers 4.42) does not run under
transformers 5.x in its cache path, and its FlashAttention-2 path cannot handle packed (remove-padding)
sequences: with attention_mask=None it calls `flash_attn_func`, which would let packed sequences
attend to each other. veRL's remove-padding path feeds exactly that layout (1, total_tokens).

This module re-implements only the features Nanbeige4.2-3B actually enables in its config.json:
  * Llama-style decoder (GQA, RoPE, SwiGLU, RMSNorm), no biases, no qk-norm
  * `num_loops` passes over the same physical layer stack (weights shared across loops)
  * the final `norm` applied at the end of every loop when `skip_loop_final_norm` is False
Everything else (ngram embeddings, hyper-connections, depth attention, loop-shared KV) raises.
Parameter names are identical to the official checkpoint, so weights load unchanged and sync to
SGLang (which has its own native Nanbeige implementation) by name.

Attention uses FlashAttention-4 (CuTe DSL, Blackwell sm_100/sm_103) varlen kernels with
cu_seqlens derived from position_ids (a new sequence starts wherever position_ids == 0), or from
attention_mask when padding is present. Training only: no KV cache, use_cache is forced off.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_nanbeige import NanbeigeConfig

_UNSUPPORTED_FLAGS = (
    "qk_layernorm", "enable_hyper_connection", "enable_mhc", "enable_depth_attention", "loop_share_kv",
    "enable_double_loop_split", "ngram_insert_all_layers", "mlp_bias", "attention_bias",
)

_FA4 = None

import os as _os

_COMPILE = _os.environ.get("KSAT_FUSE_ELEMENTWISE", "1") == "1"


def _maybe_compile(fn):
    # Inductor fuses these memory-bound chains (fp32 RMSNorm, RoPE, SwiGLU) into single Triton kernels;
    # they were ~40% of fwd+bwd time unfused. Same arithmetic/precision as the eager versions.
    return torch.compile(fn, dynamic=True) if _COMPILE else fn


@_maybe_compile
def _rms_norm(hidden_states, weight, eps: float):
    input_dtype = hidden_states.dtype
    hs = hidden_states.to(torch.float32)
    hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + eps)
    return weight * hs.to(input_dtype)


@_maybe_compile
def _swiglu(g, u):
    return F.silu(g) * u


@_maybe_compile
def _apply_rope(q, k, cos, sin):
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    return (q * c) + (torch.cat((-q2, q1), dim=-1) * s), (k * c) + (torch.cat((-k2, k1), dim=-1) * s)


def _fa4_varlen():
    global _FA4
    if _FA4 is None:
        from flash_attn.cute.interface import flash_attn_varlen_func

        _FA4 = flash_attn_varlen_func
    return _FA4


class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """Same arithmetic as the official NanbeigeRotaryEmbedding (fp32 angles, cast to model dtype).

    inv_freq is computed on the fly instead of living in a non-persistent buffer, so it survives
    meta-device init / FSDP sharding untouched.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=position_ids.device).float() / head_dim))
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        freqs = position_ids.reshape(-1, 1).float() * inv_freq.reshape(1, -1)  # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    return cos.to(dtype), sin.to(dtype)


def _cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """position_ids: (T,) flattened. New sequence wherever pos == 0 (and at index 0)."""
    T = position_ids.numel()
    starts = (position_ids == 0).nonzero(as_tuple=True)[0]
    if starts.numel() == 0 or starts[0].item() != 0:
        starts = torch.cat([torch.zeros(1, dtype=starts.dtype, device=starts.device), starts])
    cu = torch.cat([starts, torch.tensor([T], dtype=starts.dtype, device=starts.device)]).to(torch.int32)
    max_len = int((cu[1:] - cu[:-1]).max().item())
    return cu, max_len


class NanbeigeAttention(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, cu_seqlens, max_seqlen):
        # hidden_states: (T, H) packed tokens
        T = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)
        q, k = _apply_rope(q, k, cos, sin)
        if getattr(self.config, "_train_attn_backend", "fa4") == "sdpa":
            out = self._sdpa_varlen(q, k, v, cu_seqlens)
        else:
            out = _fa4_varlen()(
                q.contiguous(), k.contiguous(), v.contiguous(),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scaling, causal=True,
            )
            if isinstance(out, tuple):
                out = out[0]
        return self.o_proj(out.reshape(T, self.num_heads * self.head_dim))

    def _sdpa_varlen(self, q, k, v, cu_seqlens):
        """Reference path (slow, per-sequence SDPA) used only for numerical checks."""
        rep = self.num_heads // self.num_key_value_heads
        outs = []
        cu = cu_seqlens.tolist()
        for a, b in zip(cu[:-1], cu[1:]):
            qi = q[a:b].transpose(0, 1).unsqueeze(0)
            ki = k[a:b].repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0)
            vi = v[a:b].repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=self.scaling)
            outs.append(oi.squeeze(0).transpose(0, 1))
        return torch.cat(outs, dim=0)


class NanbeigeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class NanbeigeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NanbeigeAttention(config, layer_idx)
        self.mlp = NanbeigeMLP(config)
        self.input_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, cu_seqlens, max_seqlen=0, **kwargs):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class NanbeigeLayerStack(nn.ModuleList):
    """The physical decoder layers (state-dict names stay `model.layers.{i}.*`) plus the loop schedule.

    Runs `num_loops` passes over the same layers; `norm` (owned by NanbeigeModel) is applied after every
    loop unless skip_loop_final_norm, then once at the end. Having a forward() makes this module a valid
    FSDP2 unit that executes exactly once per model forward.
    """

    def forward(self, h, cos, sin, cu_seqlens, max_seqlen, norm, num_loops, skip_final):
        for _loop in range(num_loops):
            for layer in self:
                h = layer(h, cos, sin, cu_seqlens, max_seqlen=max_seqlen)
            if not skip_final:
                h = norm(h)
        if skip_final:
            h = norm(h)
        return h


class NanbeigePreTrainedModel(PreTrainedModel):
    config_class = NanbeigeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # Decoder layers run num_loops times per forward. Wrapping each layer as its own FSDP unit would
    # reshard/unshard and reduce-scatter per call; instead the whole decoder stack (NanbeigeLayerStack,
    # called once per forward and not containing the embedding) is one FSDP unit. 4B params fit
    # unsharded on a B300, so nothing is lost in memory.
    _no_split_modules = ["NanbeigeLayerStack"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [r"rotary_emb\.inv_freq"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, NanbeigeRMSNorm):
            module.weight.data.fill_(1.0)


class NanbeigeModel(NanbeigePreTrainedModel):
    def __init__(self, config: NanbeigeConfig):
        super().__init__(config)
        for flag in _UNSUPPORTED_FLAGS:
            if getattr(config, flag, False):
                raise NotImplementedError(f"modeling_nanbeige_train does not implement config.{flag}=True")
        if getattr(config, "ngram_vocab_size_ratio", None) or getattr(config, "insert_ngram_layer_idx", None):
            raise NotImplementedError("ngram embeddings are not implemented in modeling_nanbeige_train")
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = NanbeigeLayerStack([NanbeigeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, S, H = inputs_embeds.shape
        device = inputs_embeds.device

        # Flatten to packed layout (T, H) and derive varlen boundaries.
        indices = None
        if attention_mask is not None and not bool(attention_mask.all()):
            mask = attention_mask.bool()
            indices = mask.reshape(-1).nonzero(as_tuple=True)[0]
            lens = mask.sum(-1).to(torch.int32)
            cu_seqlens = F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
            max_seqlen = int(lens.max().item())
            if position_ids is None:
                position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            pos = position_ids.reshape(-1)[indices]
            h = inputs_embeds.reshape(B * S, H)[indices]
        else:
            if position_ids is None:
                position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
            if B == 1:
                cu_seqlens, max_seqlen = _cu_seqlens_from_position_ids(position_ids.reshape(-1))
            else:  # unpadded batch: every row is one full sequence
                cu_seqlens = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=device)
                max_seqlen = S
            pos = position_ids.reshape(-1)
            h = inputs_embeds.reshape(B * S, H)

        cos, sin = _rope_cos_sin(pos, self.layers[0].self_attn.head_dim, self.config.rope_theta, h.dtype)
        h = self.layers(h, cos, sin, cu_seqlens, max_seqlen, self.norm,
                        max(int(getattr(self.config, "num_loops", 1)), 1),
                        bool(getattr(self.config, "skip_loop_final_norm", False)))

        if indices is not None:
            out = torch.zeros(B * S, H, dtype=h.dtype, device=device)
            out[indices] = h
            h = out
        h = h.reshape(B, S, H)
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=None)


class NanbeigeForCausalLM(NanbeigePreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.model = NanbeigeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=outputs.last_hidden_state)
