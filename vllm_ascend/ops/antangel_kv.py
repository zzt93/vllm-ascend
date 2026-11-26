"""AntAngel BF16 MLA on A3: keep each kernel block inside its manager page."""
import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_dtype_size

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def uses_antangel_kv(config):
    hf = config.model_config.hf_text_config
    return (
        getattr(hf, "model_type", None) == "bailing_hybrid"
        and "BailingMoeV3ForCausalLM" in (getattr(hf, "architectures", None) or ())
        and getattr(hf, "q_lora_rank", None) is None
        and config.model_config.dtype == torch.bfloat16
        and config.quant_config is None
        and get_ascend_device_type() == AscendDeviceType.A3
    )


@triton.jit
def _scatter_component(rows, slots, cache, block_size: tl.constexpr, block_stride: tl.constexpr,
                       row_stride: tl.constexpr, width: tl.constexpr, capacity: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    if (slot >= 0) & (slot < capacity):
        cols = tl.arange(0, width)
        row = tl.load(rows + token * width + cols)
        offset = (slot // block_size) * block_stride + (slot % block_size) * row_stride
        tl.store(cache + offset + cols, row)


def scatter_component(rows, slots, cache):
    blocks, block_size, heads, width = cache.shape
    assert heads == 1 and cache.stride(-1) == 1 and rows.is_contiguous()
    _scatter_component[(rows.shape[0],)](
        rows, slots.contiguous(), cache, block_size, cache.stride(0), cache.stride(1), width, blocks * block_size
    )



def _reshape_combined_mla_kv_cache(
    raw_cache: torch.Tensor,
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    dtype: torch.dtype,
    page_stride_bytes: int,
    num_blocks_per_kv_block: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place both MLA components in each page's kernel slots, preserving shared-pool page addresses."""
    if len(k_shape) != 4 or len(v_shape) != 4 or k_shape[:3] != v_shape[:3]:
        raise ValueError("Combined MLA cache components must have matching block/token/head dimensions.")
    dtype_size = get_dtype_size(dtype)
    if num_blocks_per_kv_block < 1:
        raise ValueError("Combined MLA cache requires a positive kernel-block ratio.")
    if page_stride_bytes <= 0 or page_stride_bytes % (num_blocks_per_kv_block * dtype_size):
        raise ValueError("Combined MLA pages must split into dtype-aligned kernel slots.")
    slot_stride_bytes = page_stride_bytes // num_blocks_per_kv_block
    k_block_bytes = math.prod(k_shape[1:]) * dtype_size
    v_block_bytes = math.prod(v_shape[1:]) * dtype_size
    if k_block_bytes + v_block_bytes > slot_stride_bytes:
        raise ValueError("Combined MLA kernel slot is smaller than its K/V components.")
    if k_shape[0] * slot_stride_bytes > raw_cache.numel() * raw_cache.element_size():
        raise ValueError("Combined MLA kernel slots exceed the layer's raw cache slice.")
    storage_offset_bytes = raw_cache.storage_offset() * raw_cache.element_size()
    if storage_offset_bytes % dtype_size:
        raise ValueError("Combined MLA raw cache offset must be dtype-aligned.")
    typed_cache = raw_cache.view(dtype)

    def component_view(shape, offset_bytes):
        dense_strides = [math.prod(shape[dim + 1 :]) for dim in range(len(shape))]
        return torch.as_strided(
            typed_cache,
            size=shape,
            stride=(slot_stride_bytes // dtype_size, *dense_strides[1:]),
            storage_offset=(storage_offset_bytes + offset_bytes) // dtype_size,
        )

    return component_view(k_shape, 0), component_view(v_shape, k_block_bytes)
