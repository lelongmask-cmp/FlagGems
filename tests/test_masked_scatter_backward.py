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
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    sizes = shape

    ref_grad = utils.to_reference(grad_output)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten.masked_scatter_backward(ref_grad, ref_mask, sizes)
    with flag_gems.use_gems():
        res_out = flag_gems.masked_scatter_backward(grad_output, mask, sizes)

    utils.gems_assert_equal(res_out, ref_out)
