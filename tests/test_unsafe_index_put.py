import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise", reason="Issues #3836: To Fix (Runtime Or LLVM)"
)


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# Test shapes: (input_shape, [indices_shapes...], values_shape, accumulate)
UNSAFE_INDEX_PUT_SHAPES = (
    # 1D input, 1 index
    ((32,), ((8,),), (8,), False),
    ((100,), ((100,),), (100,), True),
    # 2D input, 2 indices
    ((32, 32), ((8,), (8,)), (8,), False),
    ((32, 32), ((8,), (2, 8)), (8,), False),
    ((32, 32), ((2, 8),), (32,), False),
    # 3D input, 3 indices
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), False),
    # 1D accumulate
    ((100,), ((100,),), (100,), True),
    # 2D accumulate: 2D index into first dim, (32, 32) + suffix (32,)
    ((32, 32), ((32, 32),), (32, 32, 32), True),
    # 1D index into first dim with suffix dims
    ((16, 16, 4), ((16,),), (16, 16, 4), False),
)


def gen_indices_for_unsafe_index_put(input_shape, indices_shapes, accumulate):
    """Generate multi-dimensional integer index tensors."""
    indices = []
    for i, shape in enumerate(indices_shapes):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.unsafe_index_put
@pytest.mark.parametrize(
    "input_shape, indices_shapes, values_shape, accumulate",
    UNSAFE_INDEX_PUT_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index_put(input_shape, indices_shapes, values_shape, accumulate, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_unsafe_index_put(
        input_shape, indices_shapes, accumulate
    )
    values = torch.randn(
        values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    # Reference: PyTorch native (without FlagGems dispatch)
    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, accumulate)

    # FlagGems result
    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, accumulate)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("dtype", [torch.float32])
def test_unsafe_index_put_negative_indices(dtype):
    """Verify negative index wrap-around behavior."""
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [torch.tensor([-1, -5, -10], device=flag_gems.device)]
    values = torch.randn((3, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_functional():
    """Verify that the input tensor is NOT modified (functional semantics)."""
    inp = torch.randn((32, 32), device=flag_gems.device)
    inp_copy = inp.clone()
    indices = [torch.randint(0, 32, (8,), device=flag_gems.device)]
    values = torch.randn((8, 32), device=flag_gems.device)

    with flag_gems.use_gems():
        out = torch._unsafe_index_put(inp, indices, values, False)

    assert torch.equal(inp, inp_copy), "Input tensor was modified"
    assert not torch.equal(out, inp), "Output should differ from input"


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_bool_mask():
    """Verify bool mask fallback works correctly."""
    inp = torch.randn((16, 16), device=flag_gems.device)
    mask = torch.randint(0, 2, (16,), dtype=torch.bool, device=flag_gems.device)
    K = mask.sum().item()
    indices = [mask]
    values = torch.randn((K, 16), device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(mask)]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_scalar_value():
    """Verify scalar value is correctly broadcast."""
    inp = torch.randn((16, 16), device=flag_gems.device)
    indices = [torch.tensor([0, 5, 10], device=flag_gems.device)]
    values = torch.tensor(3.14, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_accumulate():
    """Verify accumulate mode (atomic add)."""
    inp = torch.randn((32, 32), device=flag_gems.device)
    indices = [torch.randint(0, 32, (64,), device=flag_gems.device)]
    values = torch.randn((64, 32), device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, True)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, True)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
