"""
InfiniteContext-1B — Attention Basics (Educational Module)
==========================================================

This module implements attention from first principles so you understand
exactly what MLA will improve upon. Each class is heavily commented with
the math and intuition.

Concepts covered:
  1. Scaled Dot-Product Attention  — the core operation
  2. Multi-Head Attention (MHA)    — standard transformer attention
  3. KV Cache                      — how autoregressive decoding works
  4. Memory measurement            — see the KV cache bottleneck

Reading order:
  Start with ScaledDotProductAttention, then MultiHeadAttention,
  then CachedMultiHeadAttention. Each builds on the previous one.

Reference:
  "Attention Is All You Need" (Vaswani et al., 2017)
  https://arxiv.org/abs/1706.03762
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# ============================================================================
#
# This is the fundamental operation in every transformer.
#
# Given:
#   Q (queries)  — "what am I looking for?"    shape: (B, L, D)
#   K (keys)     — "what do I contain?"         shape: (B, S, D)
#   V (values)   — "what information do I hold?" shape: (B, S, D)
#
# Attention computes:
#   1. Similarity scores:  scores = Q @ K^T / sqrt(D)
#   2. Normalize:          weights = softmax(scores)
#   3. Weighted sum:        output = weights @ V
#
# Why scale by sqrt(D)?
#   Without scaling, as D grows large, the dot products grow large too,
#   pushing softmax into regions with tiny gradients. Scaling keeps the
#   variance of dot products at ~1 regardless of dimension.


def scaled_dot_product_attention(
    query: torch.Tensor,   # (B, ..., L, D)
    key: torch.Tensor,     # (B, ..., S, D)
    value: torch.Tensor,   # (B, ..., S, D)
    mask: torch.Tensor | None = None,  # Boolean/0-1 keep mask; (L, S) or broadcastable
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Math:
        Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    This is a FUNCTION, not a module — it has no learnable parameters.
    The learnable parts are in the projections (see MultiHeadAttention).
    """
    d_k = query.size(-1)
    scale = math.sqrt(d_k)

    # Step 1: Compute attention scores
    # Q @ K^T -> (B, ..., L, S)
    # Each query token gets a score against every key token
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    # Step 2: Apply causal mask (for autoregressive / decoder models)
    # The mask prevents attending to future tokens
    if mask is not None:
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0/1 (or False/True)")
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # Empty rows contribute zero, without creating NaNs in the backward pass.
        empty_rows = ~(mask != 0).any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty_rows, 0)

    # Step 3: Softmax normalizes scores to probabilities
    # Each row sums to 1.0 — it's a weighted average
    weights = F.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(empty_rows, 0)

    # Step 4: Weighted sum of values
    # output[i] = sum(weights[i,j] * value[j] for j in range(S))
    output = torch.matmul(weights, value)

    return output


# ============================================================================
# 2. MULTI-HEAD ATTENTION (MHA)
# ============================================================================
#
# Key insight: Instead of one big attention operation, we split Q, K, V
# into multiple "heads" that each attend to different aspects of the input.
#
# For example, with d_model=512 and n_heads=8:
#   - Each head gets d_head = 512/8 = 64 dimensions
#   - Head 1 might learn to attend to syntax
#   - Head 2 might learn to attend to semantics
#   - Head 3 might learn positional relationships
#   - etc.
#
# THE KV CACHE PROBLEM:
#   During autoregressive generation, for each new token we generate, we
#   need the K and V from ALL previous tokens. So we cache them.
#
#   Cache size per layer = 2 × B × L × n_heads × d_head × sizeof(dtype)
#                        = 2 × B × L × d_model × sizeof(dtype)
#
#   For a 1B model (d_model=2048, 24 layers) at 128k context in FP16:
#     = 2 × 1 × 131072 × 2048 × 2 bytes × 24 layers
#     = ~25.8 GB  ← This is why your RTX 2070 Super (8GB) would OOM!
#
#   MLA solves this by compressing K and V into a shared latent vector,
#   reducing cache by ~93%. That's Phase 3 of this project.


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention as described in "Attention Is All You Need."

    This is the BASELINE that MLA improves upon. Understanding this module
    is essential before implementing MLA's compression.

    Architecture:
        Input → [W_Q, W_K, W_V] → Split Heads → Attention → Merge Heads → W_O → Output
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model:  Model dimension (e.g., 512, 2048). Must be divisible by n_heads.
            n_heads:  Number of attention heads (e.g., 8, 32).
        """
        super().__init__()

        if (type(d_model) is not int or type(n_heads) is not int
                or d_model <= 0 or n_heads <= 0 or d_model % n_heads):
            raise ValueError("d_model and n_heads must be positive integers, with d_model divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per head

        # Projection matrices — these are the LEARNABLE parameters
        # W_Q, W_K, W_V: (d_model → d_model) each
        # W_O: merges heads back to d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # (B, L, d_model) — input sequence
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Steps:
            1. Project input to Q, K, V
            2. Reshape to separate heads
            3. Compute attention per head
            4. Merge heads and project output
        """
        self._validate_input(x)
        B, L, _ = x.shape

        # Step 1: Linear projections
        # Each produces (B, L, d_model), which we'll split into heads
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Step 2: Reshape to (B, n_heads, L, d_head)
        # einops makes this readable: "batch seq (heads dim) -> batch heads seq dim"
        Q = rearrange(Q, "b l (h d) -> b h l d", h=self.n_heads)
        K = rearrange(K, "b l (h d) -> b h l d", h=self.n_heads)
        V = rearrange(V, "b l (h d) -> b h l d", h=self.n_heads)

        # Step 3: Scaled dot-product attention (per head, in parallel)
        output = scaled_dot_product_attention(Q, K, V, mask=mask)
        # output shape: (B, n_heads, L, d_head)

        # Step 4: Merge heads back to (B, L, d_model)
        output = rearrange(output, "b h l d -> b l (h d)")

        # Step 5: Final linear projection
        output = self.W_O(output)

        return output

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.d_model or not x.shape[0] or not x.shape[1]:
            raise ValueError("x must have shape (positive batch, positive sequence, d_model)")

    def count_kv_cache_bytes(self, batch_size: int, seq_len: int, dtype=torch.float16) -> int:
        """
        Calculate how many bytes the KV cache would use for this layer.

        This is the number MLA dramatically reduces.

        Formula: 2 (K+V) × batch × seq_len × n_heads × d_head × bytes_per_element
        """
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        return 2 * batch_size * seq_len * self.n_heads * self.d_head * bytes_per_element


# ============================================================================
# 3. CACHED MULTI-HEAD ATTENTION (for autoregressive generation)
# ============================================================================
#
# During text generation, we produce tokens one at a time:
#   "The" → "The cat" → "The cat sat" → ...
#
# Without a cache, at step N we'd recompute K and V for ALL N tokens.
# That's O(N²) redundant work.
#
# With a KV cache, we store K and V from previous steps and only compute
# the new token's K and V. This makes generation O(N) per step.
#
# But the cache GROWS with every token — that's the memory bottleneck.


class CachedMultiHeadAttention(MultiHeadAttention):
    """
    Multi-Head Attention with KV caching for autoregressive generation.

    This demonstrates the memory problem that MLA solves:
    the KV cache grows linearly with sequence length.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 8192):
        super().__init__(d_model, n_heads)
        if type(max_seq_len) is not int or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len

        # Cache holds only the consumed prefix, up to max_seq_len.
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None
        self.cache_len = 0  # How many tokens are currently cached

    def reset_cache(self):
        """Clear the KV cache (call at start of each new generation)."""
        self.cache_k = None
        self.cache_v = None
        self.cache_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with KV caching.

        During generation:
          - x has shape (B, 1, d_model) — just the new token
          - We append its K, V to the cache
          - Q attends to the full cached K, V

        Args:
            x: (B, L, d_model) where L=1 during generation, L=full during prefill
        """
        self._validate_input(x)
        B, L, _ = x.shape
        total_len = self.cache_len + L
        if total_len > self.max_seq_len:
            raise ValueError("request exceeds max_seq_len; reset the cache before a new sequence")

        # Project the new token(s)
        Q = rearrange(self.W_Q(x), "b l (h d) -> b h l d", h=self.n_heads)
        K_new = rearrange(self.W_K(x), "b l (h d) -> b h l d", h=self.n_heads)
        V_new = rearrange(self.W_V(x), "b l (h d) -> b h l d", h=self.n_heads)

        # Form the new prefix before committing it, so failed calls keep the old cache.
        if self.cache_k is None:
            keys, values = K_new, V_new
        else:
            if (self.cache_k.shape[0] != B or self.cache_k.device != K_new.device
                    or self.cache_k.dtype != K_new.dtype):
                raise ValueError("batch, device, or dtype changed; reset the cache first")
            # ponytail: concatenation copies the prefix; use fixed buffers if profiling warrants it.
            keys = torch.cat([self.cache_k, K_new], dim=2)
            values = torch.cat([self.cache_v, V_new], dim=2)

        # Attend to ALL cached keys and values
        # Q: (B, H, L_new, D)  K_cached: (B, H, L_total, D)
        query_positions = self.cache_len + torch.arange(L, device=x.device)
        key_positions = torch.arange(total_len, device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        output = scaled_dot_product_attention(Q, keys, values, mask=causal)

        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.W_O(output)

        self.cache_k, self.cache_v, self.cache_len = keys, values, total_len
        return output

    def get_cache_memory_bytes(self) -> int:
        """Return current KV cache memory usage in bytes."""
        if self.cache_k is None:
            return 0
        # K cache + V cache
        return self.cache_k.nelement() * self.cache_k.element_size() + \
               self.cache_v.nelement() * self.cache_v.element_size()


# ============================================================================
# 4. UTILITY: Create causal mask
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention mask.

    Each token can only attend to itself and previous tokens.
    This is what makes a model "autoregressive" (can't peek at the future).

    Example for seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


# ============================================================================
# 5. MEMORY ANALYSIS HELPER
# ============================================================================

def analyze_kv_cache_scaling(
    d_model: int = 2048,
    n_heads: int = 16,
    n_layers: int = 24,
    dtype: torch.dtype = torch.float16,
    context_lengths: list[int] | None = None
) -> list[dict]:
    """
    Calculate KV cache memory for standard MHA at various context lengths.

    This describes cache tensors only. Weights, activations, temporary attention
    scores, and allocator overhead require additional memory.

    Returns a list of dicts with {seq_len, cache_mb, cache_gb} for each length.
    """
    if context_lengths is None:
        context_lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072]

    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    d_head = d_model // n_heads
    results = []

    for seq_len in context_lengths:
        # Per layer: 2 (K+V) × batch(1) × seq_len × n_heads × d_head × bytes
        per_layer = 2 * 1 * seq_len * n_heads * d_head * bytes_per_element
        total = per_layer * n_layers
        results.append({
            "seq_len": seq_len,
            "cache_bytes": total,
            "cache_mb": total / (1024**2),
            "cache_gb": total / (1024**3),
        })

    return results
