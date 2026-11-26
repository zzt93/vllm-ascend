# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)
# SPDX-License-Identifier: Apache-2.0
"""Native dim-last causal conv1d state update for Ascend GDN/KDA layers.

The upstream CUDA kernel assumes a ``(lines, dim, history)`` state and a
``(dim, width)`` weight. GDN layers on Ascend keep the state dim-last
(``[lines, history, dim]``) with a ``[width, dim]`` weight, and the PyTorch
fallback that handles that layout calls ``.item()`` once per request, which
stalls ACL graph capture. This module implements the same arithmetic directly
on the dim-last layout without any host synchronisation.
"""

import os

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

_DEBUG = os.environ.get("VLLM_ASCEND_DEBUG_CONV1D") == "1"


def _dbg(msg):
    if _DEBUG:
        print("[conv1d-npu] " + msg, flush=True)

try:  # triton-ascend ships triton; keep the import lazy-ish for unit tests.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - only on hosts without triton
    triton = None
    tl = None
    _HAS_TRITON = False

# The kernel unrolls one iteration per speculative token. Larger prefill
# windows fall back to the reference implementation.
MAX_STATIC_TOKENS = 8


if _HAS_TRITON:

    @triton.jit
    def _conv1d_update_dimlast_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        st_ptr,
        idx_ptr,
        acc_ptr,
        qsl_ptr,
        out_ptr,
        dim,
        x_stride_tok,
        x_stride_dim,
        w_stride_k,
        w_stride_dim,
        st_stride_line,
        st_stride_hist,
        st_stride_dim,
        out_stride_tok,
        out_stride_dim,
        state_len,
        start_token,
        WIDTH: tl.constexpr,
        WIDTH_P2: tl.constexpr,
        MAX_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        SILU: tl.constexpr,
        HAS_ACC: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NULL_ID: tl.constexpr,
    ):
        i = tl.program_id(0)
        pid_d = tl.program_id(1)

        line = tl.load(idx_ptr + i)
        if line == NULL_ID:
            return

        if IS_VARLEN:
            start = tl.load(qsl_ptr + i)
            end = tl.load(qsl_ptr + i + 1)
            n_tok = end - start
        else:
            start = start_token + i
            n_tok = 1
        if HAS_ACC:
            acc = tl.load(acc_ptr + i)
            n_tok = tl.minimum(n_tok, acc)
        if n_tok <= 0:
            return

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < dim

        offs_k = tl.arange(0, WIDTH_P2)
        mask_k = offs_k < WIDTH
        weight = tl.load(
            w_ptr + offs_k[:, None] * w_stride_k + offs_d[None, :] * w_stride_dim,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        else:
            bias = tl.zeros([BLOCK_D], dtype=tl.float32)

        st_base = st_ptr + line * st_stride_line + offs_d * st_stride_dim
        x_base = x_ptr + offs_d * x_stride_dim
        out_base = out_ptr + offs_d * out_stride_dim

        # For token ``t`` the conv window covers concat(history, tokens) entries
        # [state_len - WIDTH + 1 + t, state_len + t]. Entries below state_len come
        # from the state, the rest from the new tokens.
        for t in tl.static_range(MAX_T):
            live = t < n_tok
            src = state_len - WIDTH + 1 + t + offs_k
            from_state = src < state_len
            h_idx = tl.minimum(src, state_len - 1)
            h_idx = tl.maximum(h_idx, 0)
            tok_idx = src - state_len

            v_state = tl.load(
                st_base + h_idx[:, None] * st_stride_hist,
                mask=mask_d[None, :] & from_state[:, None],
                other=0.0,
            ).to(tl.float32)
            v_x = tl.load(
                x_base + (start + tl.maximum(tok_idx, 0))[:, None] * x_stride_tok,
                mask=mask_d[None, :] & (tok_idx[:, None] >= 0) & (tok_idx[:, None] < n_tok),
                other=0.0,
            ).to(tl.float32)
            window = tl.where(from_state[:, None], v_state, v_x)
            window = tl.where(mask_k[:, None], window, 0.0)
            y = tl.sum(weight * window, axis=0) + bias
            if SILU:
                y = y * tl.sigmoid(y)
            tl.store(
                out_base + (start + t) * out_stride_tok,
                y.to(out_ptr.dtype.element_ty),
                mask=mask_d & live,
            )

        # New state: the last (WIDTH - 1) entries of concat(history, tokens).
        j = tl.arange(0, WIDTH_P2)
        live_j = j < (WIDTH - 1)
        src_j = n_tok + j
        from_state_j = src_j < (WIDTH - 1)
        h_j = tl.maximum(tl.minimum(src_j, WIDTH - 2), 0)
        tok_j = src_j - (WIDTH - 1)
        v_state_j = tl.load(
            st_base + h_j[:, None] * st_stride_hist,
            mask=mask_d[None, :] & from_state_j[:, None],
            other=0.0,
        )
        v_x_j = tl.load(
            x_base + (start + tl.maximum(tok_j, 0))[:, None] * x_stride_tok,
            mask=mask_d[None, :] & (tok_j[:, None] >= 0) & (tok_j[:, None] < n_tok),
            other=0.0,
        )
        new_state = tl.where(from_state_j[:, None], v_state_j, v_x_j)
        tl.store(
            st_base + j[:, None] * st_stride_hist,
            new_state,
            mask=mask_d[None, :] & live_j[:, None],
        )


def _as_dim_last_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return the weight as ``[width, dim]`` (a view when possible)."""
    if weight.shape[0] <= weight.shape[1]:
        return weight
    return weight.transpose(0, 1)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = 1,
    null_block_id: int | None = None,
    pad_slot_id: int | None = None,
    validate_data: bool = False,
    out: torch.Tensor | None = None,
    **kwargs,
):
    """Update the GDN conv state in place on the dim-last layout.

    Mirrors the upstream signature; unsupported shapes fall back to the
    reference implementation in :mod:`vllm_ascend.ops.causal_conv1d`.
    """
    from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as _ref

    pad_id = null_block_id if null_block_id is not None else (pad_slot_id if pad_slot_id is not None else PAD_SLOT_ID)

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    if x.dim() == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if not _HAS_TRITON or x.dim() != 2 or conv_state.dim() != 3:
        _dbg("fallback ndim x=%s state=%s" % (x.dim(), conv_state.dim()))
        return _ref(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_id,
        )

    dim = x.shape[-1]
    # The state may arrive dim-last (lines, hist, dim) or dim-second
    # (lines, dim, hist); the reference implementation accepts both, so pick the
    # history/dim axes from the actual shape and let the kernel use strides.
    if conv_state.shape[-1] == dim:
        hist_axis, dim_axis = 1, 2
    elif conv_state.shape[-2] == dim:
        hist_axis, dim_axis = 2, 1
    else:
        _dbg("fallback state layout %s dim=%d" % (tuple(conv_state.shape), dim))
        return _ref(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_id,
        )

    # Weight may arrive [width, dim] or [dim, width]; the window axis is the one
    # that is not the feature dim.
    if weight.shape[1] == dim:
        width = weight.shape[0]
        w_stride_k, w_stride_dim = weight.stride(0), weight.stride(1)
    elif weight.shape[0] == dim:
        width = weight.shape[1]
        w_stride_k, w_stride_dim = weight.stride(1), weight.stride(0)
        weight = weight.transpose(0, 1)
    else:
        _dbg("fallback weight layout %s dim=%d" % (tuple(weight.shape), dim))
        return _ref(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_id,
        )
    if width > 8:
        _dbg("fallback width=%d" % width)
        return _ref(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_id,
        )

    if query_start_loc is not None:
        hint = int(max_query_len)
        tokens = hint if hint > 0 else 1
    else:
        tokens = 1
    if tokens <= 0 or tokens > MAX_STATIC_TOKENS:
        _dbg("fallback tokens=%d shape=%s" % (tokens, tuple(x.shape)))
        return _ref(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_id,
        )

    batch = conv_state_indices.numel() if conv_state_indices is not None else x.shape[0]
    if batch == 0:
        return x
    state_len = width - 1
    out = out if out is not None else torch.empty_like(x)

    idx = conv_state_indices
    if idx is None:
        idx = torch.arange(batch, dtype=torch.int32, device=x.device)

    grid = (batch, triton.cdiv(dim, 128))
    _conv1d_update_dimlast_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        conv_state,
        idx,
        num_accepted_tokens if num_accepted_tokens is not None else idx,
        query_start_loc if query_start_loc is not None else idx,
        out,
        dim,
        x.stride(0),
        x.stride(1),
        w_stride_k,
        w_stride_dim,
        conv_state.stride(0),
        conv_state.stride(hist_axis),
        conv_state.stride(dim_axis),
        out.stride(0),
        out.stride(1),
        state_len,
        0,
        WIDTH=width,
        WIDTH_P2=triton.next_power_of_2(width),
        MAX_T=max(tokens, 1),
        BLOCK_D=128,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        HAS_ACC=num_accepted_tokens is not None,
        IS_VARLEN=query_start_loc is not None,
        NULL_ID=pad_id,
    )
    return out
