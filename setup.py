# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# build clib
import os
_ext_src_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clib")
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
    name='clib',
    ext_modules=[
        CUDAExtension(
            name='clib._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    entry_points={
        'console_scripts': [
            'fairnr-train = train:main'
        ],
    },
)