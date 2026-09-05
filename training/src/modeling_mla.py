"""DeepSeek-style latent KV attention with decoupled RoPE and a dense reference.

The reference and absorbed paths share weights. This implements a configurable
MLA decoder architecture; it does not supply pretrained weights or claim novelty.
Method reference: https://github.com/deepseek-ai/DeepSeek-V2
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MLAConfig:
    vocab_size: int = 50257
    hidden_size: int = 1536
    num_layers: int = 32
    intermediate_size: int = 4352
    num_heads: int = 16
    kv_rank: int = 256
    q_rank: int = 512
    nope_dim: int = 64
    rope_dim: int = 64
    value_dim: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        dimensions = (self.vocab_size, self.hidden_size, self.num_layers, self.intermediate_size,
                      self.num_heads, self.kv_rank, self.q_rank, self.nope_dim, self.rope_dim, self.value_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError('all model dimensions must be positive integers')
        if self.rope_dim % 2 or not math.isfinite(self.rope_theta) or self.rope_theta <= 1 or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('RoPE dimension must be even, theta > 1, and normalization epsilon positive')


@dataclass(frozen=True)
class MLACache:
    latent: torch.Tensor
    rope_keys: torch.Tensor
    total_tokens: int

    @property
    def length(self):
        return self.latent.shape[1]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.latent, self.rope_keys))

    def retain(self, limit, anchors):
        if self.length <= limit:
            return self
        if limit < anchors or anchors < 0:
            raise ValueError('cache limit must leave space for the anchors')
        tail = limit - anchors
        indices = torch.cat((torch.arange(anchors, device=self.latent.device),
                             torch.arange(self.length - tail, self.length, device=self.latent.device)))
        return MLACache(self.latent[:, indices], self.rope_keys[:, indices], self.total_tokens)


def rotate(x, positions, theta):
    dimension = x.shape[-1]
    precision = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequencies = theta ** (-torch.arange(0, dimension, 2, device=x.device, dtype=precision) / dimension)
    angles = positions.to(precision)[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    half = dimension // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + rotated * sine


class MLAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.q_a_proj = nn.Linear(c.hidden_size, c.q_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(c.q_rank, eps=c.norm_eps)
        self.q_b_proj = nn.Linear(c.q_rank, c.num_heads * (c.nope_dim + c.rope_dim), bias=False)
        self.kv_a_proj = nn.Linear(c.hidden_size, c.kv_rank + c.rope_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(c.kv_rank, eps=c.norm_eps)
        self.kv_b_proj = nn.Linear(c.kv_rank, c.num_heads * (c.nope_dim + c.value_dim), bias=False)
        self.o_proj = nn.Linear(c.num_heads * c.value_dim, c.hidden_size, bias=False)

    def forward(self, x, cache=None, *, use_cache=False, implementation='absorbed', window=None, anchors=4):
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.hidden_size or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError('input must be a nonempty (batch, sequence, hidden_size) tensor')
        if implementation not in ('absorbed', 'reference'):
            raise ValueError('implementation must be absorbed or reference')
        if (cache is not None or window is not None) and not use_cache:
            raise ValueError('cache state and window policy require use_cache=True')
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError('cached decoding requires eval mode and disabled gradients')
        batch, length, _ = x.shape
        if window is not None:
            if type(window) is not int or type(anchors) is not int or anchors < 0 or length > window - anchors:
                raise ValueError('the window must hold anchors and the entire input chunk')
        if cache is not None:
            if cache.latent.shape != (batch, cache.length, c.kv_rank) or cache.rope_keys.shape != (batch, cache.length, c.rope_dim):
                raise ValueError('cache dimensions do not match the input/model')
            if cache.latent.device != x.device or cache.rope_keys.device != x.device or type(cache.total_tokens) is not int or cache.total_tokens < cache.length:
                raise ValueError('cache device or token count is inconsistent')
            if window is not None:
                cache = cache.retain(window - length, anchors)
        previous = 0 if cache is None else cache.length
        total = length if cache is None else cache.total_tokens + length
        queries = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        queries = queries.view(batch, length, c.num_heads, c.nope_dim + c.rope_dim).transpose(1, 2)
        query_content, query_rope = queries.split((c.nope_dim, c.rope_dim), dim=-1)
        compressed, key_rope = self.kv_a_proj(x).split((c.kv_rank, c.rope_dim), dim=-1)
        latent = self.kv_a_norm(compressed)
        if cache is not None:
            if cache.latent.dtype != latent.dtype or cache.rope_keys.dtype != key_rope.dtype:
                raise ValueError('cache dtype changed; start a new cache')
            latent = torch.cat((cache.latent, latent), dim=1)
            key_rope = torch.cat((cache.rope_keys, key_rope), dim=1)
        # Raw rotary keys permit rebasing without repeatedly rotating stored tensors.
        rotated_keys = rotate(key_rope, torch.arange(latent.shape[1], device=x.device), c.rope_theta)
        rotated_queries = rotate(query_rope, previous + torch.arange(length, device=x.device), c.rope_theta)
        weights = self.kv_b_proj.weight.view(c.num_heads, c.nope_dim + c.value_dim, c.kv_rank)
        key_up, value_up = weights.split((c.nope_dim, c.value_dim), dim=1)
        if implementation == 'reference':
            keys = torch.einsum('bkr,hdr->bhkd', latent, key_up)
            values = torch.einsum('bkr,hvr->bhkv', latent, value_up)
            scores = torch.einsum('bhqd,bhkd->bhqk', query_content, keys)
        else:
            absorbed_query = torch.einsum('bhqd,hdr->bhqr', query_content, key_up)
            scores = torch.einsum('bhqr,bkr->bhqk', absorbed_query, latent)
        scores = (scores + torch.einsum('bhqd,bkd->bhqk', rotated_queries, rotated_keys)) / math.sqrt(c.nope_dim + c.rope_dim)
        causal = torch.arange(latent.shape[1], device=x.device)[None, :] <= (previous + torch.arange(length, device=x.device))[:, None]
        scores = scores.masked_fill(~causal, float('-inf'))
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype).to(x.dtype)
        if implementation == 'reference':
            output = torch.einsum('bhqk,bhkv->bhqv', probabilities, values)
        else:
            latent_output = torch.einsum('bhqk,bkr->bhqr', probabilities, latent)
            output = torch.einsum('bhqr,hvr->bhqv', latent_output, value_up)
        output = output.transpose(1, 2).reshape(batch, length, c.num_heads * c.value_dim)
        state = MLACache(latent, key_rope, total) if use_cache else None
        return self.o_proj(output), state


class MLADecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = MLAAttention(config)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, cache=None, **options):
        attention, state = self.attention(self.attention_norm(x), cache, **options)
        x = x + attention
        normalized = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), state


class MLALanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MLADecoderLayer(config) for _ in range(config.num_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, caches=None, *, use_cache=False, implementation='absorbed', window=None, anchors=4):
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.numel():
            raise ValueError('input_ids must be a nonempty two-dimensional integer tensor')
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError('token ID is outside the configured vocabulary')
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError('one cache is required per decoder layer')
        if caches and any((cache.length, cache.total_tokens) != (caches[0].length, caches[0].total_tokens) for cache in caches):
            raise ValueError('decoder-layer cache positions must agree')
        x = self.embedding(input_ids)
        states = []
        for index, layer in enumerate(self.layers):
            x, state = layer(x, None if caches is None else caches[index], use_cache=use_cache,
                             implementation=implementation, window=window, anchors=anchors)
            if use_cache:
                states.append(state)
        logits = self.lm_head(self.norm(x))
        return (logits, states) if use_cache else logits
