# FlagGems — Claude Configuration

## Project Overview

FlagGems is a high-performance, generic operator library implemented in Triton language. It provides backend-neutral kernels for accelerating LLM training and inference across diverse hardware platforms (NVIDIA, Ascend, Cambricon, Kunlunxin, etc.). Operators register with the PyTorch ATen backend for seamless integration.

## Project Layout

```
src/flag_gems/ops/       - Operator implementations (one file per op)
                           e.g., masked_scatter_backward.py
tests/                    - Accuracy tests (pytest, one file per op)
                           e.g., test_masked_scatter_backward.py
benchmark/                - Standalone benchmark scripts
                           e.g., test_masked_scatter_backward.py (benchmark variant)
src/flag_gems/runtime/    - Runtime backend support
src/flag_gems/utils/      - Shared utilities
src/flag_gems/__init__.py - Operator registration / public API
pyproject.toml            - Project metadata and dependencies
```

## Adding a New Operator

1. **Op implementation** — `src/flag_gems/ops/<op_name>.py`
   - Wrap Triton kernels with PyTorch autograd function or direct callable
   - Use Apache 2.0 license header
   - Import `torch` and other flag_gems ops as needed
   - Register via `__init__.py` exports

2. **Tests** — `tests/test_<op_name>.py`
   - Use pytest with `@pytest.mark.<op_name>` marker
   - Compare against `torch.ops.aten.<op_name>` reference
   - Use `accuracy_utils.gems_assert_equal` for comparison
   - Test across multiple shapes, dtypes, and edge cases

3. **Benchmark** — `benchmark/test_<op_name>.py`
   - Standalone script comparing flag_gems vs torch native perf
   - Use CUDA events for timing

## Key Conventions

- All code uses Apache 2.0 license header
- Triton kernels are written in `triton_src/` or inline in op files
- Tests use `flag_gems.use_gems()` context manager to enable gems ops
- Tests use `utils.to_reference()` to move tensors to reference device
- Device is obtained via `flag_gems.device`

## Build / Test Commands

```bash
# Install in editable mode
pip install -e .

# Run a specific test
pytest tests/test_masked_scatter_backward.py -v

# Run all tests for a specific op marker
pytest -m masked_scatter_backward -v

# Run benchmark
python benchmark/test_masked_scatter_backward.py
```
