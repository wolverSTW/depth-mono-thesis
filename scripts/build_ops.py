"""Build original CUDA sources with the actual GPU architecture, no upstream edits."""
import os
from pathlib import Path
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parents[1]
if not torch.cuda.is_available() or CUDA_HOME is None:
    raise RuntimeError("Build on Linux CUDA with CUDA toolkit/nvcc installed")
source = ROOT / "MonoIA/lib/models/monoia/ops/src"
sources = list(source.glob("*.cpp")) + list((source / "cpu").glob("*.cpp")) + list((source / "cuda").glob("*.cu"))
os.chdir(ROOT)
setup(name="depth-monoia-upstream-ops", version="0.1", packages=[],
      ext_modules=[CUDAExtension("MultiScaleDeformableAttention", [str(p) for p in sources],
                  include_dirs=[str(source)], define_macros=[("WITH_CUDA", None)],
                  extra_compile_args={"cxx": [], "nvcc": ["-DCUDA_HAS_FP16=1",
                      "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                      "-D__CUDA_NO_HALF2_OPERATORS__"]})],
      cmdclass={"build_ext": BuildExtension})
