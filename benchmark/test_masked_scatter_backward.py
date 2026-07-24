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


# TODO(Qiming): Move this to an abstraction layer
class TensorSelectBackwardBenchmark(base.GenericBenchmark2DOnly):
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


def _input_fn(shape, cur_dtype, device):
    grad_output = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3
    sizes = shape

    yield grad_output, mask, sizes


def _get_gbps(bench_fn_args, latency):
    grad_output, mask, sizes = bench_fn_args
    io_amount = sum(
        [shape_utils.size_in_bytes(item) for item in [grad_output, mask, grad_output]]
    )

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_scatter_backward
def test_masked_scatter_backward():
    bench = TensorSelectBackwardBenchmark(
        op_name="masked_scatter_backward",
        torch_op=torch.ops.aten.masked_scatter_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()
