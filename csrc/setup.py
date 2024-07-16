
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess

setup(
    name='fused_convs',
    ext_modules=[
        CUDAExtension('fused_convs', [
            'main.cpp',
            'convs/convs_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                             'nvcc': ['-O3', '-lineinfo', '--use_fast_math', '-std=c++17', '--ptxas-options=-v',
                             '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF_OPERATORS__',
                             '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                             '--threads', '2']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='0.0.0',
    description='fused whisper conv',
    url='https://github.com/saeeddhqan/fused_convs',
    author='Saeed',
    license='Apache 2.0',
)
