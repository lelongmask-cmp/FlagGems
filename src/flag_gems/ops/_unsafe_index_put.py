import logging

import torch

from flag_gems import config

logger = logging.getLogger(__name__)


def _unsafe_index_put(inp, indices, values, accumulate=False):
    """_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    Functional advanced indexing scatter. Returns a new tensor.
    All parameter forms (bool masks, None indices, basic+advanced mixed) are
    handled in the C++ wrapper; no Python fallback needed.
    """
    logger.debug("GEMS _UNSAFE_INDEX_PUT")

    if not indices:
        raise ValueError("At least one index tensor is required")

    # The C++ wrapper's Triton kernel handles accumulate for all dtypes:
    # - float32, float64, int32, int64 → native tl.atomic_add
    # - float16, bfloat16, int16 → CAS-based atomic add
    # - int8, uint8 → cast to int32, accumulate, cast back
    if config.has_c_extension:
        try:
            from flag_gems import c_operators

            return c_operators.unsafe_index_put(inp, indices, values, accumulate)
        except ImportError:
            pass

    # Pure Python fallback (should rarely be reached)
    indices = list(indices)
    out = inp.clone()
    torch.ops.aten._index_put_impl_(out, indices, values, accumulate, unsafe=True)
    return out
