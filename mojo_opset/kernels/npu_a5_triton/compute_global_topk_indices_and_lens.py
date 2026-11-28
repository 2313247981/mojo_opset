"""A5 Triton block-table lookup with fused valid-entry counting; forward only."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

# One program owns a block of consecutive tokens plus one K tile. DAV_3510 has
# 248KB of UB per vector core; the live int32 tiles (local index, block id,
# offset, slot and the gather index) stay near 96KB so the rest of the tile
# remains on chip. The runtime also expands each entry with the opposite
# multibuffer setting.
_K_TILES = (512, 1024, 2048, 4096)
_UB_TILE_BYTES = 96 * 1024
_BYTES_PER_INDEX = 24


def _block_configs():
    configs = []
    for k_tile in _K_TILES:
        rows = _UB_TILE_BYTES // (k_tile * _BYTES_PER_INDEX)
        rows = 1 << (rows.bit_length() - 1)
        configs.append(triton.Config({"TRITON_BLOCK_SIZE": k_tile, "TOKEN_BLOCK": rows}))
    return configs


@triton.autotune(
    configs=_block_configs(),
    key=["topk"],
)
@libentry()
@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,
    num_tokens,
    TRITON_BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    max_blocks: tl.constexpr,
):
    token_ids = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    token_mask = token_ids < num_tokens
    is_valid_token = tl.load(is_valid_token_ptr + token_ids, mask=token_mask, other=0)
    req_idx = tl.load(token_to_req_indices_ptr + token_ids, mask=token_mask, other=0)


    row_cols = tl.arange(0, MAX_BLOCKS)
    table_rows = tl.load(
        block_table_ptr + req_idx[:, None] * block_table_stride + row_cols[None, :],
        mask=token_mask[:, None] & (row_cols[None, :] < max_blocks),
        other=0,
    ).to(tl.float32)

    count = tl.zeros((TOKEN_BLOCK,), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        k_mask = offset < topk
        tile_mask = token_mask[:, None] & k_mask[None, :]

        local_idx = tl.load(
            topk_indices_ptr + token_ids[:, None] * topk_indices_stride + offset[None, :],
            mask=tile_mask,
            other=-1,
        )
        is_valid = local_idx >= 0

        block_indices = local_idx // block_size
        # -1 is out of range for the gather. The later tl.where only rewrites
        # the result, so the index has to be in range before the gather runs.
        safe_idx = tl.where(is_valid, block_indices, 0)
        block_numbers = tl.gather(table_rows, safe_idx, 1).to(tl.int32)
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_ids[:, None] * global_topk_indices_stride + offset[None, :],
            slot_ids,
            mask=tile_mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=1)

    # Zero out length for padding tokens. Tokens past num_tokens stay unstored.
    tl.store(
        topk_lens_ptr + token_ids,
        tl.where(is_valid_token, count, 0),
        mask=token_mask,
    )


@torch.library.custom_op("mojo_npu_triton_a5::compute_global_topk_indices_and_lens", mutates_args=())
def compute_global_topk_indices_and_lens_fwd(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.contiguous()
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    if num_tokens == 0:
        return global_topk_indices, topk_lens
    max_blocks = block_table.shape[-1]

    def grid(meta):
        return (triton.cdiv(num_tokens, meta["TOKEN_BLOCK"]),)

    _compute_global_topk_indices_and_lens_kernel[grid](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        num_tokens,
        MAX_BLOCKS=triton.next_power_of_2(max_blocks),
        max_blocks=max_blocks,
    )
    return global_topk_indices, topk_lens


@compute_global_topk_indices_and_lens_fwd.register_fake
def _compute_global_topk_indices_and_lens_fake(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(topk_indices, memory_format=torch.contiguous_format),
        topk_indices.new_empty((topk_indices.shape[0],), dtype=torch.int32),
    )
