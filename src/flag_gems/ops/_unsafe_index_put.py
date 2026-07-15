import logging

import torch

from flag_gems import config

logger = logging.getLogger(__name__)


def _unsafe_index_put(inp, indices, values, accumulate=False):
    """_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    Functional advanced indexing scatter. Returns a new tensor.
    Equivalent to: out = self.clone(); out[indices] = values (or += values if accumulate=True).
    "unsafe" means no bounds checking assertion (but negative indices are still wrapped).
    """
    logger.debug("GEMS _UNSAFE_INDEX_PUT")

    if not indices:
        raise ValueError("At least one index tensor is required")

    # Fast path: all indices are non-None LongTensors already on the input device.
    # Skip preprocessing overhead for the common case.
    all_tensor = True
    needs_preprocess = False
    for idx in indices:
        if idx is None:
            all_tensor = False
            break
        if idx.dtype in (torch.bool, torch.int8):
            needs_preprocess = True
            break
        if idx.device != inp.device:
            needs_preprocess = True
            break

    if all_tensor and not needs_preprocess and config.has_c_extension:
        try:
            from flag_gems import c_operators

            return c_operators.unsafe_index_put(inp, indices, values, accumulate)
        except (RuntimeError, ImportError):
            pass

    # Slow path: preprocessing needed (bool/int8 expansion, None padding, device transfer)
    indices = list(indices)

    # Device transfer for indices
    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]

    # Expand bool/byte masks into explicit integer index tensors
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            expanded = idx.nonzero(as_tuple=True)
            processed_indices.extend(expanded)
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices

    if len(indices) > inp.ndim:
        raise IndexError(
            "too many indices for tensor of dimension {}".format(inp.ndim)
        )

    # Try C++ with preprocessed indices (only non-None tensors)
    tensor_indices = [idx for idx in indices if idx is not None]

    if tensor_indices and config.has_c_extension:
        try:
            from flag_gems import c_operators

            return c_operators.unsafe_index_put(
                inp, tensor_indices, values, accumulate
            )
        except (RuntimeError, ImportError):
            pass

    # Final fallback for None-padding cases or C++ failure
    if len(indices) < inp.ndim:
        indices = indices + [None] * (inp.ndim - len(indices))

    out = inp.clone()
    torch.ops.aten._index_put_impl_(out, indices, values, accumulate, unsafe=True)
    return out
