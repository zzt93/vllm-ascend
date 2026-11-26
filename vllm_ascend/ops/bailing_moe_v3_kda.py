# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import wraps

import torch
from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd
from fla_npu.ops.ascendc import recurrent_kda
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.ops.gdn import causal_conv1d_fn
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend, AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops.triton.kda.kda import chunk_kda
from vllm_ascend.ops.triton.kda.utils import FLA_CHUNK_SIZE
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_SUPPORTED_CONV_WIDTHS = (2, 3, 4)
_CONV_CHANNEL_ALIGNMENT = 16
_ASCENDC_KDA_HEAD_DIM = 128
_PACKED_DECODE_WEIGHT = "_packed_decode_weight"


@dataclass
class _ChunkMetadata:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor


@dataclass
class _NativeChunkMetadata:
    cu_seqlens: tuple[int, ...]
    live_rows: torch.Tensor | None


def _prepare_native_chunk_metadata(
    query_offsets: tuple[int, ...], device: torch.device
) -> _NativeChunkMetadata | None:
    # The AscendC entry point expects strictly increasing Host offsets.
    # Empty requests have no token slice but still occupy a state row.
    live_rows = [row for row, (a, b) in enumerate(zip(query_offsets, query_offsets[1:])) if a < b]
    if not live_rows:
        return None
    if len(live_rows) == len(query_offsets) - 1:
        return _NativeChunkMetadata(query_offsets, None)
    compact_offsets = (query_offsets[0], *(query_offsets[row + 1] for row in live_rows))
    return _NativeChunkMetadata(
        compact_offsets, torch.tensor(live_rows, dtype=torch.int64).to(device, non_blocking=True)
    )


def _prepare_chunk_metadata(query_offsets: tuple[int, ...], device: torch.device) -> _ChunkMetadata:
    # Preserve the physical row number even if an empty request appears in
    # the middle of the sequence list.
    indices = []
    offsets = [0]
    for row, (start, end) in enumerate(zip(query_offsets, query_offsets[1:])):
        chunk_count = (end - start + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
        indices.extend((row, chunk) for chunk in range(chunk_count))
        offsets.append(offsets[-1] + chunk_count)
    return _ChunkMetadata(
        torch.tensor(indices, dtype=torch.int32).reshape(-1, 2).to(device, non_blocking=True),
        torch.tensor(offsets, dtype=torch.int32).to(device, non_blocking=True),
    )


class _BailingMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> GDNAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
            num_actual_reqs,
        )
        prefill = metadata.non_spec_prefill_metadata
        if prefill is not None:
            starts = metadata.non_spec_query_start_loc
            assert starts is not None
            offsets = prefill.chunk.cu_seqlens_host
            if len(offsets) != starts.numel():
                # The base builder omits decode rows from its chunk metadata.
                # Bailing runs the entire non-speculative batch through chunk KDA.
                offsets = tuple(common_attn_metadata.query_start_loc_cpu[: starts.numel()].tolist())
            if offsets == prefill.chunk.cu_seqlens_host and all(a < b for a, b in zip(offsets, offsets[1:])):
                metadata.bailing_chunk_metadata = prefill.chunk
            else:
                metadata.bailing_chunk_metadata = _prepare_chunk_metadata(offsets, starts.device)
            if get_ascend_device_type() == AscendDeviceType.A3:
                metadata.bailing_native_chunk_metadata = _prepare_native_chunk_metadata(offsets, starts.device)
        return metadata


class _BailingAttentionBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[_BailingMetadataBuilder]:
        return _BailingMetadataBuilder


def _active_rows(
    indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_size: int,
) -> torch.Tensor:
    return (indices > NULL_BLOCK_ID) & (indices < cache_size) & (query_start_loc[1:] > query_start_loc[:-1])


def _prefill_convolution(
    projections: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    biases: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    dim_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = _active_rows(indices, query_start_loc, state.shape[0])
    # The physical cache may be page-strided and may differ from the model
    # dtype. Pack only the active request rows into the native op's SD layout.
    state_rows = gather_ssm_states(state, indices, active, output_dtype=projections[0].dtype)
    if dim_first:
        state_rows = state_rows.transpose(-1, -2).contiguous()
    compact_indices = torch.arange(indices.numel(), device=indices.device, dtype=indices.dtype)
    compact_indices = torch.where(active, compact_indices, PAD_SLOT_ID)

    mixed_qkv = torch.cat(projections, dim=-1)
    packed_weight = torch.cat(weights, dim=0).transpose(0, 1).contiguous().to(mixed_qkv.dtype)
    packed_bias = None
    if any(bias is not None for bias in biases):
        packed_bias = torch.cat(
            [weight.new_zeros(weight.shape[0]) if bias is None else bias for weight, bias in zip(weights, biases)]
        ).to(mixed_qkv.dtype)

    result = causal_conv1d_fn(
        mixed_qkv,
        packed_weight,
        packed_bias,
        conv_states=state_rows,
        query_start_loc=query_start_loc,
        cache_indices=compact_indices,
        has_initial_state=(has_initial_state & active).contiguous(),
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        # Compact row 0 is a real request. Invalid rows use the pad sentinel.
        null_block_id=PAD_SLOT_ID,
    )

    state_len = weights[0].shape[1] - 1
    # Leave any extra speculative history columns untouched, including their
    # precision if conv_state has a different dtype than the native op.
    if state_rows.shape[1] != state_len:
        original_rows = gather_ssm_states(state, indices, active)
        if dim_first:
            original_rows[..., :state_len].copy_(state_rows[:, :state_len].transpose(-1, -2))
        else:
            original_rows[:, :state_len].copy_(state_rows[:, :state_len])
        state_rows = original_rows
    elif dim_first:
        state_rows = state_rows.transpose(-1, -2).contiguous()

    scatter_ssm_states_(state, indices, state_rows, active_rows=active)
    q, k, v = result.split([projection.shape[-1] for projection in projections], dim=-1)
    return q, k, v


def _register_packed_decode_weight(conv: torch.nn.Module) -> None:
    """Keep a width-major copy for decode without changing checkpoint weights."""
    conv.register_buffer(_PACKED_DECODE_WEIGHT, None, persistent=False)
    process_weights = conv.quant_method.process_weights_after_loading
    weight_loader = conv.weight.weight_loader

    @torch.no_grad()
    def refresh() -> None:
        packed = getattr(conv, _PACKED_DECODE_WEIGHT)
        weight = conv.weight
        if weight.is_meta:
            return
        arranged = weight.squeeze(1).transpose(0, 1)
        if packed is not None and (packed.shape, packed.device, packed.dtype) == (
            arranged.shape,
            arranged.device,
            arranged.dtype,
        ):
            # Decode graphs retain the buffer address across weight reloads.
            packed.copy_(arranged)
        else:
            setattr(conv, _PACKED_DECODE_WEIGHT, arranged.contiguous())

    @wraps(process_weights)
    def process_and_pack(*args, **kwargs):
        result = process_weights(*args, **kwargs)
        refresh()
        return result

    @wraps(weight_loader)
    def load_and_repack(*args, **kwargs):
        result = weight_loader(*args, **kwargs)
        if getattr(conv, _PACKED_DECODE_WEIGHT) is not None:
            refresh()
        return result

    conv.quant_method.process_weights_after_loading = process_and_pack
    conv.weight.weight_loader = load_and_repack


class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):
    """Reuse the upstream projection, decode and speculative paths on Ascend."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            _register_packed_decode_weight(conv)

    def _get_decode_conv_weights(
        self, weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = getattr(self.q_conv1d, _PACKED_DECODE_WEIGHT, None)
        k = getattr(self.k_conv1d, _PACKED_DECODE_WEIGHT, None)
        v = getattr(self.v_conv1d, _PACKED_DECODE_WEIGHT, None)
        if q is None or k is None or v is None:
            return weights
        return q, k, v

    def get_attn_backend(self) -> type[_BailingAttentionBackend]:
        return _BailingAttentionBackend

    def _run_prefill_convolution(
        self,
        projections: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        conv_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_state = self.kv_cache[0]
        indices = metadata.non_spec_state_indices_tensor
        query_start_loc = metadata.non_spec_query_start_loc
        has_initial_state = metadata.has_initial_state
        assert indices is not None and query_start_loc is not None and has_initial_state is not None
        if (
            conv_state.numel() == 0
            or not conv_state[0].is_contiguous()
            or weights[0].shape[-1] not in _SUPPORTED_CONV_WIDTHS
            or sum(p.shape[-1] for p in projections) % _CONV_CHANNEL_ALIGNMENT != 0
            or projections[0].dtype not in (torch.float16, torch.bfloat16)
        ):
            return super()._run_prefill_convolution(projections, weights, conv_states, metadata)

        return _prefill_convolution(
            projections,
            weights,
            (self.q_conv1d.bias, self.k_conv1d.bias, self.v_conv1d.bias),
            conv_state,
            indices,
            has_initial_state,
            query_start_loc,
            dim_first=is_conv_state_dim_first(),
        )

    def _run_recurrent_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            q.dtype != torch.bfloat16
            or q.shape[-1] != 128
            or v.shape[-1] != 128
            or state.dtype != torch.float32
            or state_indices.ndim != 1
            or state_indices.numel() < q.shape[1]
        ):
            return super()._run_recurrent_decode(q, k, v, gate, beta, state, query_start_loc, state_indices)

        # The gate and beta have already been computed by the Bailing layer.
        # Pass the original cache view so the op reads the physical page stride.
        out, _ = recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            state,
            cu_seqlens=query_start_loc,
            ssm_state_indices=state_indices,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            output_final_state=False,
            inplace_final_state=True,
        )
        return out

    def _run_chunk_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        recurrent_state_active: torch.Tensor,
        metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        indices = metadata.non_spec_state_indices_tensor
        query_start_loc = metadata.non_spec_query_start_loc
        has_initial_state = metadata.has_initial_state
        assert indices is not None and query_start_loc is not None and has_initial_state is not None
        if (
            recurrent_state_active.numel() == 0
            or recurrent_state.shape != recurrent_state_active.shape
            or not recurrent_state_active[0].is_contiguous()
        ):
            return super()._run_chunk_prefill(q, k, v, gate, beta, recurrent_state, recurrent_state_active, metadata)

        active = _active_rows(indices, query_start_loc, recurrent_state_active.shape[0])
        native_meta = getattr(metadata, "bailing_native_chunk_metadata", None)
        if (
            native_meta is not None
            and q.shape == k.shape == v.shape
            and q.shape[0] == 1
            and q.shape[1] == native_meta.cu_seqlens[-1]
            and q.shape[-1] == _ASCENDC_KDA_HEAD_DIM
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and gate.dtype == beta.dtype == recurrent_state_active.dtype == torch.float32
        ):
            if native_meta.live_rows is not None:
                indices = indices.index_select(0, native_meta.live_rows)
                has_initial_state = has_initial_state.index_select(0, native_meta.live_rows)
                active = active.index_select(0, native_meta.live_rows)
            initial_state = gather_ssm_states(recurrent_state_active, indices, has_initial_state & active)
            # Use the native chunk pipeline on A3, where Triton W/U can produce
            # nonfinite results even for finite inputs. The gate is precomputed.
            out, last_state, *_ = ascendc_chunk_kda_fwd(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                gate.contiguous(),
                beta.contiguous(),
                scale=q.shape[-1] ** -0.5,
                chunk_size=FLA_CHUNK_SIZE,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=native_meta.cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                state_v_first=True,
            )
            scatter_ssm_states_(recurrent_state_active, indices, last_state, active_rows=active)
            return out

        initial_state = gather_ssm_states(recurrent_state_active, indices, has_initial_state & active)

        out, last_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=query_start_loc,
            prebuilt_meta=getattr(metadata, "bailing_chunk_metadata", None),
        )
        scatter_ssm_states_(recurrent_state_active, indices, last_state, active_rows=active)
        return out
