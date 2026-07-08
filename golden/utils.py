# Original license:
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

"""
A collection of utilities for ensuring that training can always occur. Heavily influenced by the
[toma](https://github.com/BlackHC/toma) library.
"""

import functools
import gc
import inspect
import warnings
import torch
import importlib
from functools import lru_cache
import os
import importlib.metadata
from packaging.version import parse, Version
from packaging import version
from typing import Union
import operator as op

torch_version = parse(importlib.metadata.version("torch"))

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


def split_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_lists(lst, ids, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n], ids[i:i + n]


def get_best_device() -> str:
    """
    Returns the best available device string in priority order:
    Apple Metal (MPS) > CUDA > CPU.
    """
    if is_mps_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def clear_torch_cache() -> None:
    """Free memory caches for whichever accelerator is active."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS (Apple Metal) cache clearing — available from PyTorch 2.0+
    if is_mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib.metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(torch_version, operation, version)


def str_to_bool(value) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value!r}")


def parse_flag_from_env(key, default=False):
    """Returns truthy value for `key` from the env if available else the default."""
    value = os.environ.get(key, str(default))
    return str_to_bool(value) == 1  # As its name indicates `str_to_bool` actually returns an int...


def is_mps_available() -> bool:
    """Returns True if Apple Metal (MPS) is available and built into this PyTorch install."""
    return (
        is_torch_version(">=", "1.12")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    "check if user disables it explicitly"
    if not parse_flag_from_env("ACCELERATE_USE_XPU", default=True):
        return False
    "Checks if `intel_extension_for_pytorch` is installed and potentially if a XPU is in the environment"
    if is_ipex_available():
        import torch

        if is_torch_version("<=", "1.12"):
            return False
    else:
        return False

    import intel_extension_for_pytorch  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


# From accelerate/utils/memory.py
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to an out-of-memory condition on CUDA, MPS, CUDNN, or CPU.

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",                          # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",     # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        "MPS backend out of memory",                   # Apple Metal OOM
        "not enough memory on mps",                    # Alternative MPS OOM phrasing
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def should_reduce_batch_size_but_handle_error(exception: Exception) -> bool:
    """
    Checks if `exception` relates to a CUDA handle error caused by an excessively large batch size.

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to
    out-of-memory or CUDNN, the batch size is cut in half and passed to `function`.

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from accelerate.utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        elif is_mps_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    if is_xpu_available():
                        torch.xpu.empty_cache()
                    elif is_npu_available():
                        torch.npu.empty_cache()
                    elif is_mps_available() and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    batch_size //= 2
                elif should_reduce_batch_size_but_handle_error(e):
                    raise RuntimeError(
                        f"Batch size {batch_size} caused a nonrecoverable cuBLAS handle creation error, "
                        "likely caused by an excessively large batch size. "
                        "Try reducing the batch size slightly and trying again."
                    )
                else:
                    raise

    return decorator