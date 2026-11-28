"""Tests for functions.compute_global_topk_indices_and_lens."""

import pytest
import torch

from mojo_opset import functions as F

# The kernel walks K in TRITON_BLOCK_SIZE steps, so the sweep covers a single
# column, both sides of each 1024 boundary, and the four-tile width.
TOPK_SWEEP = [0, 1, 2, 7, 8, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 4096]
TOKEN_SWEEP = [0, 1, 2, 7, 56, 57, 127, 128, 1023, 4096]
BLOCK_SIZE_SWEEP = [1, 7, 32, 64, 96, 128, 256]


def make_inputs(
    num_tokens,
    topk,
    num_reqs,
    block_size,
    max_blocks,
    seed,
    *,
    valid_ratio=0.7,
    pad_ratio=0.15,
):
    """Build a legal int32 case on CPU.

    Local indices stay inside ``[0, max_blocks * block_size)`` before masking, so
    ``local // block_size`` always addresses a real block-table column. Physical
    block ids stay small enough that ``block_id * block_size`` fits in int32.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    span = max_blocks * block_size
    local = torch.randint(0, span, (num_tokens, topk), generator=generator, dtype=torch.int32)
    if num_tokens > 0 and topk > 0:
        if valid_ratio <= 0.0:
            local.fill_(-1)
        elif valid_ratio < 1.0:
            local = local.masked_fill(torch.rand(num_tokens, topk, generator=generator) > valid_ratio, -1)
    req = torch.randint(0, num_reqs, (num_tokens,), generator=generator, dtype=torch.int32)
    is_valid = torch.rand(num_tokens, generator=generator) >= pad_ratio
    table = torch.randint(0, 4096, (num_reqs, max_blocks), generator=generator, dtype=torch.int32)
    return local, req, table, is_valid


def reference(topk_indices, token_to_req, block_table, block_size, is_valid_token):
    """CPU int64 golden of the block-table lookup, narrowed back to int32."""
    local = topk_indices.to(torch.int64)
    req = token_to_req.to(torch.int64)
    table = block_table.to(torch.int64)
    valid_token = is_valid_token.to(torch.bool)

    valid = local >= 0
    blocks = torch.div(local, block_size, rounding_mode="floor")
    safe = torch.where(valid, blocks, torch.zeros_like(blocks))
    gathered = table[req.unsqueeze(-1), safe]
    slots = gathered * block_size + torch.remainder(local, block_size)
    slots = torch.where(valid, slots, torch.full_like(gathered, -1))

    counted = valid.sum(dim=-1)
    lens = torch.where(valid_token, counted, torch.zeros_like(counted))
    return slots.to(torch.int32), lens.to(torch.int32)


def assert_exact(actual, expected, name):
    """L0 integer rule: identical dtype, shape and every element equal."""
    assert actual.dtype == torch.int32 and expected.dtype == torch.int32
    assert tuple(actual.shape) == tuple(expected.shape)
    a = actual.detach().to("cpu", torch.int64)
    e = expected.detach().to("cpu", torch.int64)
    different = int((a != e).sum().item())
    assert different == 0, f"{name}: {different} of {a.numel()} elements differ"


def check(accuracy_backend, topk_indices, token_to_req, block_table, block_size, is_valid):
    implementation, _, device = accuracy_backend
    expected_slots, expected_lens = reference(topk_indices, token_to_req, block_table, block_size, is_valid)

    actual_slots, actual_lens = F.compute_global_topk_indices_and_lens(
        topk_indices.to(device),
        token_to_req.to(device),
        block_table.to(device),
        block_size,
        is_valid.to(device),
        implementation=implementation,
    )

    assert actual_slots.device.type == device
    assert actual_lens.device.type == device
    assert_exact(actual_slots, expected_slots, "global_topk_indices")
    assert_exact(actual_lens, expected_lens, "topk_lens")

    # Length has an oracle that never touches the slot formula.
    counted = (topk_indices >= 0).sum(dim=-1).to(torch.int32)
    counted = torch.where(is_valid.to(torch.bool), counted, torch.zeros_like(counted))
    assert_exact(actual_lens, counted, "topk_lens_count")


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_hand_checked_mapping(accuracy_backend):
    """One block boundary, one request boundary, and one padding token."""
    block_size = 64
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    topk_indices = torch.tensor([[0, 63, 64, -1], [65, -1, 1, 128]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    is_valid = torch.tensor([True, False])

    expected_slots = torch.tensor([[640, 703, 704, -1], [1345, -1, 1281, 1408]], dtype=torch.int32)
    expected_lens = torch.tensor([3, 0], dtype=torch.int32)
    reference_slots, reference_lens = reference(topk_indices, token_to_req, block_table, block_size, is_valid)
    assert_exact(reference_slots, expected_slots, "reference_slots")
    assert_exact(reference_lens, expected_lens, "reference_lens")

    check(accuracy_backend, topk_indices, token_to_req, block_table, block_size, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.parametrize("topk", TOPK_SWEEP)
@pytest.mark.accuracy
def test_topk_width_sweep(accuracy_backend, topk):
    """K sweeps both sides of every 1024-wide tile boundary."""
    topk_indices, token_to_req, block_table, is_valid = make_inputs(5, topk, 3, 64, 8, 4000 + topk)
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.parametrize("num_tokens", TOKEN_SWEEP)
@pytest.mark.accuracy
def test_token_count_sweep(accuracy_backend, num_tokens):
    """The grid follows the token count, including empty and one-past-a-core-wave."""
    topk_indices, token_to_req, block_table, is_valid = make_inputs(
        num_tokens, 2048, max(1, min(num_tokens, 32)), 64, 8, 5000 + num_tokens
    )
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.parametrize("block_size", BLOCK_SIZE_SWEEP)
@pytest.mark.accuracy
def test_block_size_sweep(accuracy_backend, block_size):
    """Page size covers 1, a non-power-of-two, and the usual powers of two."""
    topk_indices, token_to_req, block_table, is_valid = make_inputs(6, 64, 3, block_size, 4, 6000 + block_size)
    check(accuracy_backend, topk_indices, token_to_req, block_table, block_size, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.parametrize("block_size", BLOCK_SIZE_SWEEP)
@pytest.mark.accuracy
def test_local_range_edges(accuracy_backend, block_size):
    """First slot, last slot of a page, last legal index, and the sentinels."""
    max_blocks = 3
    last = max_blocks * block_size - 1
    columns = [0, block_size - 1, block_size, last, -1, -2]
    topk_indices = torch.tensor([columns, columns], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[0, 5, 9], [4, 0, 8]], dtype=torch.int32)
    is_valid = torch.tensor([True, False])
    check(accuracy_backend, topk_indices, token_to_req, block_table, block_size, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_all_invalid_indices(accuracy_backend):
    topk_indices, token_to_req, block_table, is_valid = make_inputs(
        5, 16, 2, 64, 4, 7, valid_ratio=0.0, pad_ratio=0.0
    )
    assert bool((topk_indices == -1).all())
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_duplicate_local_indices_count_twice(accuracy_backend):
    """The same local index twice is two valid entries, not one."""
    block_table = torch.tensor([[10]], dtype=torch.int32)
    topk_indices = torch.tensor([[0, 0, -1, 5]], dtype=torch.int32)
    token_to_req = torch.tensor([0], dtype=torch.int32)
    is_valid = torch.tensor([True])
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_negative_sentinels_all_become_minus_one(accuracy_backend):
    """Invalid means ``< 0``, not only the usual ``-1`` placeholder."""
    block_table = torch.tensor([[2, 3]], dtype=torch.int32)
    topk_indices = torch.tensor([[-1, -2, -100000, 0, 64]], dtype=torch.int32)
    token_to_req = torch.tensor([0], dtype=torch.int32)
    is_valid = torch.tensor([True])
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_large_physical_block_id(accuracy_backend):
    """``block_id * block_size`` stays inside int32 but is far from zero."""
    block_id = 1_000_003
    block_table = torch.tensor([[block_id, block_id + 1]], dtype=torch.int32)
    topk_indices = torch.tensor([[0, 63, 64, -1]], dtype=torch.int32)
    token_to_req = torch.tensor([0], dtype=torch.int32)
    is_valid = torch.tensor([True])
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_padding_keeps_slots_and_zeroes_length(accuracy_backend):
    """Clearing ``is_valid_token`` zeroes every length and leaves the slots."""
    implementation, _, device = accuracy_backend
    topk_indices, token_to_req, block_table, _ = make_inputs(4, 32, 2, 64, 4, 8, valid_ratio=0.8, pad_ratio=0.0)

    def run(is_valid):
        return F.compute_global_topk_indices_and_lens(
            topk_indices.to(device),
            token_to_req.to(device),
            block_table.to(device),
            64,
            is_valid.to(device),
            implementation=implementation,
        )

    slots, lens = run(torch.ones(4, dtype=torch.bool))
    slots_padded, lens_padded = run(torch.zeros(4, dtype=torch.bool))
    assert_exact(slots_padded, slots, "slots_unchanged")
    assert int(lens_padded.sum().item()) == 0
    assert lens.shape == lens_padded.shape


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_empty_tokens_and_empty_topk(accuracy_backend):
    topk_indices, token_to_req, block_table, is_valid = make_inputs(0, 0, 1, 64, 1, 9)
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_single_column_block_table(accuracy_backend):
    """Every valid local index stays inside the one page."""
    topk_indices, token_to_req, block_table, is_valid = make_inputs(7, 15, 3, 64, 1, 10, valid_ratio=0.7, pad_ratio=0.2)
    assert block_table.shape[1] == 1
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.accuracy
def test_row_stride_not_contiguous(accuracy_backend):
    """``stride(0)`` is honoured; only the last dimension has to be contiguous."""
    num_tokens, topk = 4, 16
    buffer = torch.randint(0, 256, (num_tokens * 2, topk), dtype=torch.int32)
    buffer[::2, 0] = -1
    topk_indices = buffer[::2]
    assert not topk_indices.is_contiguous()
    token_to_req = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    table_buffer = torch.randint(1, 100, (4, 8), dtype=torch.int32)
    block_table = table_buffer[::2]
    is_valid = torch.tensor([True, True, False, True])
    check(accuracy_backend, topk_indices, token_to_req, block_table, 64, is_valid)


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
def test_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton fake metadata")
    from mojo_opset import preload

    preload("compute_global_topk_indices_and_lens", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    topk_indices = torch.zeros(5, 7, dtype=torch.int32, device=device)
    token_to_req = torch.zeros(5, dtype=torch.int32, device=device)
    block_table = torch.zeros(2, 4, dtype=torch.int32, device=device)
    is_valid = torch.ones(5, dtype=torch.bool, device=device)
    torch.library.opcheck(
        ops.compute_global_topk_indices_and_lens.default,
        (topk_indices, token_to_req, block_table, 64, is_valid),
        test_utils=("test_faketensor",),
    )
