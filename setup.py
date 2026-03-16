from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def _extension_libraries() -> list[str]:
    libs = ["c10", "torch", "torch_cpu", "torch_python"]
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    names = [p.name for p in torch_lib_dir.glob("*")]
    has_c10_cuda = any("c10_cuda" in name for name in names)
    has_torch_cuda = any("torch_cuda" in name for name in names)
    if has_c10_cuda and has_torch_cuda:
        # Link CUDA libs when present so c10 CUDA symbols resolve at runtime.
        libs.extend(["c10_cuda", "torch_cuda"])
    return libs


# Build Flux as a standard Python package plus one native C++ extension.
setup(
    name="flux-profiler",
    version="0.1.0",
    description="Hardware-aware ML profiler and diagnostics",
    packages=find_packages(include=["flux", "flux.*"]),
    ext_modules=[
        CppExtension(
            name="flux._C",
            sources=["csrc/flux_hooks.cpp", "csrc/flux_cuda_hooks.cpp"],
            include_dirs=["csrc"],
            libraries=_extension_libraries(),
            # C++17 is required by modern PyTorch extension APIs.
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    entry_points={"console_scripts": ["flux=flux.cli:main"]},
    install_requires=["torch>=2.0", "numpy>=1.24"],
    python_requires=">=3.9",
)
