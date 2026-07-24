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

import logging

import torch

from flag_gems.ops.masked_select import masked_select

logger = logging.getLogger(__name__)


def masked_scatter_backward(grad_output, mask, sizes):
    """
    Backward of masked_scatter w.r.t. `source`.

    Matches aten::masked_scatter_backward(grad_output, mask, sizes) -> Tensor,
    which is registered CompositeExplicitAutograd in PyTorch, i.e. every
    backend (including ours) must supply its own kernel for it. The autograd
    formula for masked_scatter is:

        self:   grad_output.masked_fill(mask, 0)             (already covered
                                                                by our masked_fill)
        source: masked_scatter_backward(grad_output, mask, source.sizes())

    Semantics (mirrors the reference CPU/CUDA implementation exactly):
        mask_selected = grad_output.masked_select(mask)   # order-preserving
                                                            # stream compaction
        if mask_selected.numel() < prod(sizes):
            pad the tail with zeros up to prod(sizes)
        return mask_selected.view(sizes)

    The only nontrivial part of this whole op is `masked_select`, which is a
    strictly-ordered stream-compaction (exclusive-scan + scatter). We deliberately
    do *not* re-derive that primitive here: flag_gems.ops.masked_select already
    implements it as a two-kernel Triton pipeline
        (1) mask_part_sum_kernel  - per-CTA partial sums of the flattened mask,
            fused with a single-launch cross-CTA exclusive scan done via an
            atomic "last CTA turns off the lights" barrier (no 2nd host-side
            kernel launch needed for the scan step), and
        (2) write_back_kernel     - recomputes the local cumsum per CTA, adds
            the CTA's global offset, and scatter-writes elements into their
            compacted position,
    with a single-block fast path for small N (<= 4096) that just uses
    `tl.cumsum` directly. Reusing it here avoids shipping a second, subtly
    different implementation of the same scan/compaction logic.
    """
    logger.debug("GEMS MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    mask_selected = masked_select(grad_output, mask)

    diff_nelem = numel - mask_selected.numel()
    if diff_nelem > 0:
        # masked_select only returns the elements that were actually consumed
        # by `source` during the forward pass; any remaining tail of `source`
        # (when source.numel() > mask.sum()) never contributed to the output,
        # so its gradient is exactly zero.
        zeros_fillin = torch.zeros(
            diff_nelem, dtype=mask_selected.dtype, device=mask_selected.device
        )
        mask_selected = torch.cat([mask_selected, zeros_fillin], dim=0)

    return mask_selected.view(sizes)