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

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts, utils


class MaskedScatterBackwardBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def _backward_input_fn(shape, cur_dtype, device):
    # The shape the autograd engine actually passes: sizes == source shape.
    grad = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3
    numel = mask.sum().item()

    yield grad, mask, [numel]


def _backward_full_shape_input_fn(shape, cur_dtype, device):
    # sizes == grad shape: the native composite materializes zeros + cat here.
    grad = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3

    yield grad, mask, list(shape)


def _e2e_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3
    numel = mask.sum().item()
    src = utils.generate_tensor_input((numel,), cur_dtype, device)

    yield inp, mask, src


def _get_gbps(bench_fn_args, latency):
    grad, mask, sizes = bench_fn_args
    out_numel = 1
    for s in sizes:
        out_numel *= s
    io_amount = (
        shape_utils.size_in_bytes(grad)
        + shape_utils.size_in_bytes(mask)
        + out_numel * grad.element_size()
    )

    return io_amount * 1e-9 / (latency * 1e-3)


def _get_e2e_gbps(bench_fn_args, latency):
    inp, mask, src = bench_fn_args
    numel = mask.sum().item()
    # forward: clone + read mask/src + write out; backward: masked_fill
    # read/write out + this op reads grad and writes src.grad (numel elems).
    io_amount = (
        3 * shape_utils.size_in_bytes(inp)
        + shape_utils.size_in_bytes(mask)
        + 2 * numel * inp.element_size()
    )

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_scatter_backward
def test_masked_scatter_backward():
    bench = MaskedScatterBackwardBenchmark(
        op_name="masked_scatter_backward",
        torch_op=torch.ops.aten.masked_scatter_backward,
        input_fn=_backward_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()


@pytest.mark.masked_scatter_backward
def test_masked_scatter_backward_full_shape():
    bench = MaskedScatterBackwardBenchmark(
        op_name="masked_scatter_backward_full_shape",
        torch_op=torch.ops.aten.masked_scatter_backward,
        input_fn=_backward_full_shape_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()


@pytest.mark.masked_scatter_backward
def test_masked_scatter_backward_e2e():
    bench = MaskedScatterBackwardBenchmark(
        op_name="masked_scatter_backward_e2e",
        torch_op=torch.masked_scatter,
        input_fn=_e2e_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_e2e_gbps,
        is_backward=True,
    )
    bench.run()
