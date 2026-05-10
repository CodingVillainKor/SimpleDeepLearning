"""
MDLM model - patched to remove flash_attn dependency.
Uses standard PyTorch scaled_dot_product_attention instead.
"""
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange
from transformers import modeling_outputs

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


# ─── Rotary Embedding (no flash_attn) ───────────────────────
class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()  # (seq_len, dim)
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_qkv(qkv, cos, sin):
    """Apply rotary to q, k (not v). qkv: (b, s, 3, h, d)"""
    cos = cos[None, :, None, :]  # (1, s, 1, d)
    sin = sin[None, :, None, :]
    q = qkv[:, :, 0] * cos + _rotate_half(qkv[:, :, 0]) * sin
    k = qkv[:, :, 1] * cos + _rotate_half(qkv[:, :, 1]) * sin
    v = qkv[:, :, 2]
    return q, k, v


# ─── Layers ──────────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


# ─── Transformer Block (standard attention) ──────────────────
class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )

        cos, sin = rotary_cos_sin
        with torch.amp.autocast("cuda", enabled=False):
            q, k, v = apply_rotary_pos_emb_qkv(
                qkv.float(), cos.to(qkv.dtype), sin.to(qkv.dtype)
            )

        # q, k, v: (b, s, h, d) → (b, h, s, d) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x.transpose(1, 2)  # (b, s, h, d)
        x = rearrange(x, "b s h d -> b s (h d)")

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DITBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.hidden_dim,
                    config.n_heads,
                    config.cond_dim,
                    dropout=config.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output_layer = DDitFinalLayer(
            config.hidden_dim, config.vocab_size, config.cond_dim
        )

    def forward(self, indices, sigma, output_hidden_states=False):
        if not self.config.time_conditioning:
            sigma = torch.zeros_like(sigma)
        all_hidden_states = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden_states.append(x)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
                if output_hidden_states:
                    all_hidden_states.append(x)
            logits = self.output_layer(x, c)
        return logits, all_hidden_states


class MDLMConfig(transformers.PretrainedConfig):
    model_type = "mdlm"

    def __init__(
        self,
        vocab_size=50258,
        model_length=1024,
        hidden_dim=768,
        cond_dim=128,
        n_blocks=12,
        n_heads=12,
        dropout=0.1,
        time_conditioning=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.model_length = model_length
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.time_conditioning = time_conditioning


class MDLM(transformers.PreTrainedModel):
    config_class = MDLMConfig
    base_model_prefix = "mdlm"

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.backbone = DITBackbone(config)

    def forward(
        self,
        input_ids=None,
        timesteps=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if timesteps is None:
            timesteps = torch.zeros(input_ids.shape[0], device=input_ids.device)

        logits, all_hidden_states = self.backbone(
            indices=input_ids,
            sigma=timesteps,
            output_hidden_states=output_hidden_states or False,
        )

        if return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                loss=None,
            )
        return logits
