#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="teed_pointnet",
    ext_modules=[
        CUDAExtension(
            name="teed_pointnet",
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
            ],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
