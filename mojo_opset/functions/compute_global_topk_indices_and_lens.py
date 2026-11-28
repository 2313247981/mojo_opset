"""Forward-only block-table lookup for paged KV cache top-k indices."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate local top-k indices into global KV cache slots and count valid ones.

    ``topk_indices`` is ``[tokens, topk]`` int32; a local index below zero is a
    placeholder that becomes -1 and never indexes the cache. ``token_to_req_indices``
    selects a ``block_table`` row per token, and ``is_valid_token`` marks padding
    tokens, whose translated slots are written but whose length is 0.

    Returns ``(global_topk_indices [tokens, topk] int32, topk_lens [tokens] int32)``.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    _validate(topk_indices, token_to_req_indices, block_table, is_valid_token)
    _require_inference("compute_global_topk_indices_and_lens", topk_indices, block_table)
    forward, _ = load_impl("compute_global_topk_indices_and_lens", implementation)
    return forward(topk_indices, token_to_req_indices, block_table, block_size, is_valid_token)


def _validate(topk_indices, token_to_req_indices, block_table, is_valid_token):
    for name, tensor in (
        ("topk_indices", topk_indices),
        ("token_to_req_indices", token_to_req_indices),
        ("block_table", block_table),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
    if is_valid_token.dtype != torch.bool:
        raise ValueError("is_valid_token must be bool")
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be [tokens, topk]")
    if token_to_req_indices.ndim != 1 or token_to_req_indices.shape[0] != topk_indices.shape[0]:
        raise ValueError("token_to_req_indices must be [tokens]")
    if is_valid_token.ndim != 1 or is_valid_token.shape[0] != topk_indices.shape[0]:
        raise ValueError("is_valid_token must be [tokens]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be [requests, max_blocks]")
    device = topk_indices.device
    for name, tensor in (
        ("token_to_req_indices", token_to_req_indices),
        ("block_table", block_table),
        ("is_valid_token", is_valid_token),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the input device")
