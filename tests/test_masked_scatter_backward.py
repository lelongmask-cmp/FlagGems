# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    THRESHOLD_SHAPE = [(0.3, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    THRESHOLD_SHAPE = list(zip([0.3, 0.5, 0.7], utils.REDUCTION_SHAPES))

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.masked_scatter_backward
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_scatter_backward(shape, dtype, threshold):
    # sizes == source shape: the shape the autograd engine actually passes.
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    numel = mask.sum().item()
    sizes = [numel]

    ref_grad = utils.to_reference(grad)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten.masked_scatter_backward(ref_grad, ref_mask, sizes)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.masked_scatter_backward(grad, mask, sizes)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_scatter_backward
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_scatter_backward_full_shape(shape, dtype, threshold):
    # sizes == grad shape: exercises the zero-fill of the tail positions.
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    sizes = list(shape)

    ref_grad = utils.to_reference(grad)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten.masked_scatter_backward(ref_grad, ref_mask, sizes)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.masked_scatter_backward(grad, mask, sizes)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_scatter_backward
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_scatter_backward_broadcast_mask(shape, dtype, threshold):
    # mask smaller than grad_output: the op expands it to grad_output's shape.
    if len(shape) < 2:
        pytest.skip("broadcast mask requires a shape with at least 2 dims")

    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (
        torch.randn(shape[-1], dtype=dtype, device=flag_gems.device) < threshold
    )
    numel = mask.expand(shape).sum().item()
    sizes = [numel]

    ref_grad = utils.to_reference(grad)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten.masked_scatter_backward(ref_grad, ref_mask, sizes)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.masked_scatter_backward(grad, mask, sizes)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_scatter_backward
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_scatter_backward_autograd(shape, dtype, threshold):
    # End-to-end: masked_scatter forward + backward through the autograd engine.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    numel = mask.sum().item()
    src = torch.randn((numel,), dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_src = utils.to_reference(src)
    ref_out = torch.masked_scatter(ref_inp, ref_mask, ref_src)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad)

    (ref_inp_grad, ref_src_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_src), ref_grad
    )
    with flag_gems.use_gems():
        res_out = torch.masked_scatter(inp, mask, src)
        (res_inp_grad, res_src_grad) = torch.autograd.grad(
            res_out, (inp, src), out_grad
        )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inp_grad, ref_inp_grad)
    utils.gems_assert_equal(res_src_grad, ref_src_grad)
