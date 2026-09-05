"""Inference-only Triton references for normalization, RoPE and latent decoding.

These kernels require numerical and profiler checks on the target GPU before
integration. They are deliberately separate from the validated PyTorch model.
"""

import math
import os

import torch
import triton as tr
import triton.language as tl


def validate(*tensors):
    interpreter = os.environ.get('TRITON_INTERPRET') == '1'
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.requires_grad or not tensor.numel():
            raise ValueError('kernels require nonempty contiguous inference tensors')
        if tensor.device != tensors[0].device or tensor.dtype != tensors[0].dtype:
            raise ValueError('kernel inputs must share device and dtype')
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('supported dtypes are fp32, fp16 and bf16')
        if not interpreter and (not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device) < (8, 0)):
            raise ValueError('Triton GPU validation requires supported CUDA hardware (SM80+)')


@tr.jit
def _rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + columns, columns < D, 0).to(tl.float32)
    w = tl.load(W + columns, columns < D, 0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(Y + row * D + columns, x * inverse * w, columns < D)


def rms_norm(x, weight, eps=1e-6):
    validate(x, weight)
    if x.ndim < 1 or weight.shape != (x.shape[-1],) or x.shape[-1] > 16384 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid RMSNorm shape or epsilon')
    output = torch.empty_like(x)
    _rms[(x.numel() // x.shape[-1],)](x, weight, output, x.shape[-1], eps, tr.next_power_of_2(x.shape[-1]))
    return output


@tr.jit
def _rope(X, P, Y, D: tl.constexpr, THETA: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    half = D // 2
    columns = tl.arange(0, BLOCK)
    position = tl.load(P + row).to(tl.float32)
    angle = position * tl.exp(-2.0 * columns / D * tl.log(THETA))
    left = tl.load(X + row * D + columns, columns < half, 0).to(tl.float32)
    right = tl.load(X + row * D + columns + half, columns < half, 0).to(tl.float32)
    cosine, sine = tl.cos(angle), tl.sin(angle)
    tl.store(Y + row * D + columns, left * cosine - right * sine, columns < half)
    tl.store(Y + row * D + columns + half, right * cosine + left * sine, columns < half)


def rope(x, positions, theta=10000.0):
    validate(x)
    if x.ndim != 2 or x.shape[-1] % 2 or x.shape[-1] > 16384 or positions.shape != (x.shape[0],):
        raise ValueError('RoPE expects rows of even width and one position per row')
    if positions.device != x.device or not positions.is_contiguous() or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError('RoPE positions must be contiguous integer indices on the input device')
    if not math.isfinite(theta) or theta <= 1 or torch.any(positions < 0) or torch.any(positions > 32768):
        raise ValueError('RoPE requires theta > 1 and positions in the checked range 0..32768')
    output = torch.empty_like(x)
    _rope[(x.shape[0],)](x, positions, output, x.shape[-1], float(theta), tr.next_power_of_2(x.shape[-1] // 2))
    return output


@tr.jit
def _decode(Q, QR, K, KR, O, N: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
            D: tl.constexpr, SCALE: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    ranks, dims = tl.arange(0, BR), tl.arange(0, BD)
    q = tl.load(Q + (batch * H + head) * R + ranks, ranks < R, 0).to(tl.float32)
    qr = tl.load(QR + (batch * H + head) * D + dims, dims < D, 0).to(tl.float32)
    maximum, denominator = float('-inf'), 0.0
    accumulator = tl.full((BR,), 0, tl.float32)
    for start in range(tl.cdiv(N, BK)):
        offsets = start * BK + tl.arange(0, BK)
        latent = tl.load(K + (batch * N + offsets[:, None]) * R + ranks[None, :],
                         (offsets[:, None] < N) & (ranks[None, :] < R), 0).to(tl.float32)
        rotary = tl.load(KR + (batch * N + offsets[:, None]) * D + dims[None, :],
                         (offsets[:, None] < N) & (dims[None, :] < D), 0).to(tl.float32)
        scores = (tl.sum(latent * q[None, :], 1) + tl.sum(rotary * qr[None, :], 1)) * SCALE
        scores = tl.where(offsets < N, scores, float('-inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        rescale = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        accumulator = accumulator * rescale + tl.sum(probabilities[:, None] * latent, 0)
        denominator = denominator * rescale + tl.sum(probabilities, 0)
        maximum = next_maximum
    tl.store(O + (batch * H + head) * R + ranks, accumulator / denominator, ranks < R)


def mla_decode(query, query_rope, latent, rope_keys, *, scale):
    validate(query, query_rope, latent, rope_keys)
    if any(x.ndim != 3 for x in (query, query_rope, latent, rope_keys)):
        raise ValueError('latent decoding requires three-dimensional tensors')
    batch, heads, rank = query.shape
    tokens, dimension = latent.shape[1], query_rope.shape[-1]
    if latent.shape != (batch, tokens, rank) or query_rope.shape != (batch, heads, dimension) or rope_keys.shape != (batch, tokens, dimension):
        raise ValueError('latent decoding dimensions disagree')
    if max(rank, dimension) > 512 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('latent dimensions must be <=512 and scale positive')
    output = torch.empty_like(query)
    _decode[(batch, heads)](query, query_rope, latent, rope_keys, output, tokens, heads, rank,
                          dimension, float(scale), tr.next_power_of_2(rank), tr.next_power_of_2(dimension), 32)
    return output
