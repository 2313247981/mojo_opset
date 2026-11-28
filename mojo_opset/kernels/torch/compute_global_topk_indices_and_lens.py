"""Device-independent block-table lookup and valid-entry counting."""

import torch


def compute_global_topk_indices_and_lens_fwd(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    A local index below zero is a placeholder: it becomes -1 and never indexes
    the block table. A padding token still has its row translated, but reports
    length 0. The arithmetic runs in int64 so block ids near the int32 slot
    limit cannot overflow mid-computation, then narrows back to int32.
    """
    local = topk_indices.to(torch.int64)
    req = token_to_req_indices.to(torch.int64)
    table = block_table.to(torch.int64)
    valid_token = is_valid_token.to(torch.bool)

    valid = local >= 0
    block_indices = torch.div(local, block_size, rounding_mode="floor")
    safe_blocks = torch.where(valid, block_indices, torch.zeros_like(block_indices))
    gathered = table[req.unsqueeze(-1), safe_blocks]
    offsets = torch.remainder(local, block_size)
    slots = torch.where(valid, gathered * block_size + offsets, torch.full_like(gathered, -1))

    count = valid.sum(dim=-1)
    lens = torch.where(valid_token, count, torch.zeros_like(count))
    return slots.to(torch.int32).contiguous(), lens.to(torch.int32).contiguous()
