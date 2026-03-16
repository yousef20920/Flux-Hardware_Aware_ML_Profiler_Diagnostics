from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Build Flux as a standard Python package plus one native C++ extension.
setup(
    name="flux-profiler",
    version="0.1.0",
    description="Hardware-aware ML profiler and diagnostics",
    packages=find_packages(include=["flux", "flux.*"]),
    ext_modules=[
        CppExtension(
            name="flux._C",
            sources=["csrc/flux_hooks.cpp"],
            include_dirs=["csrc"],
            # C++17 is required by modern PyTorch extension APIs.
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    entry_points={"console_scripts": ["flux=flux.cli:main"]},
    install_requires=["torch>=2.0", "numpy>=1.24"],
    python_requires=">=3.9",
)
