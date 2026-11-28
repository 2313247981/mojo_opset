"""Performance cases for functions.compute_global_topk_indices_and_lens.

Each case is the full block-table lookup plus valid-entry counting in one kernel:
it reads ``topk_indices`` once and writes ``global_topk_indices`` once, so the
resident traffic is ``8 * T * K`` bytes (int32 in, int32 out). Shapes are sized
to 112-120 MiB so the working set sits past L2 and the steady state comes from
HBM, split into small / medium / large tiers with two generalization widths.

``topk`` samples the 1024-wide K tile the kernel walks: one tile, two tiles, and
four tiles. ``block_size`` 64 and 128 are the two supported page sizes and do not
change the byte count, so the second page size shares a ``(T, K)`` with the first.
"""

import pytest
import torch

from mojo_opset import functions

# (num_tokens, topk, block_size). Locals stay valid so the block-table load is
# live; padding and -1 stay in the accuracy tests. Traffic is 8 * T * K bytes.
SMALL_CASES = [
    (14336, 1024, 64),   # 112 MiB, one 1024-wide tile
    (7168, 2048, 64),    # 112 MiB, two tiles
]
MEDIUM_CASES = [
    (14848, 1024, 64),   # 116 MiB
    (7424, 2048, 64),    # 116 MiB, two tiles
    (7424, 2048, 128),   # same traffic, the other page size
]
LARGE_CASES = [
    (15360, 1024, 64),   # 120 MiB
    (7680, 2048, 64),    # 120 MiB, two tiles
    (3840, 4096, 64),    # 120 MiB, four tiles
]
GENERAL_CASES = [
    (16383, 896, 64),    # 2^14 - 1 tokens, 112 MiB
    (10240, 1536, 64),   # non-power-of-two width, 120 MiB
]

ALL_CASES = SMALL_CASES + MEDIUM_CASES + LARGE_CASES + GENERAL_CASES

MAX_BLOCKS = 32
NUM_REQS = 4


def _case_id(num_tokens, topk, block_size):
    return f"T{num_tokens}-K{topk}-BS{block_size}"


def _make(num_tokens, topk, block_size, device):
    """All-valid locals and no padding, so the block-table load stays live."""
    generator = torch.Generator()
    generator.manual_seed(42)
    span = MAX_BLOCKS * block_size
    topk_indices = torch.randint(
        0, span, (num_tokens, topk), generator=generator, dtype=torch.int32
    ).to(device)
    token_to_req = torch.randint(
        0, NUM_REQS, (num_tokens,), generator=generator, dtype=torch.int32
    ).to(device)
    block_table = torch.randint(
        0, 4096, (NUM_REQS, MAX_BLOCKS), generator=generator, dtype=torch.int32
    ).to(device)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    # Order matches the public signature so ``*args`` cannot shift the scalars.
    return topk_indices, token_to_req, block_table, block_size, is_valid


@pytest.mark.api("functions.compute_global_topk_indices_and_lens")
@pytest.mark.parametrize(
    "num_tokens,topk,block_size", ALL_CASES, ids=[_case_id(*case) for case in ALL_CASES]
)
@pytest.mark.parametrize("phase", ["forward"])
def test_compute_global_topk_indices_and_lens(
    benchmark, perf_environment, num_tokens, topk, block_size, phase
):
    _, device, _, implementation = perf_environment

    def factory():
        args = _make(num_tokens, topk, block_size, device)
        return lambda: functions.compute_global_topk_indices_and_lens(
            *args, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="compute_global_topk_indices_and_lens",
        num_tokens=num_tokens,
        topk=topk,
        block_size=block_size,
        dtype="int32",
        phase=phase,
    )
