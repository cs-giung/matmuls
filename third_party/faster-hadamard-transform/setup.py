from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="faster_hadamard_transform",
    ext_modules=[
        CUDAExtension(
            name="faster_hadamard_transform",
            sources=[
                "hadamard_transform.cpp",
                "hadamard_transform_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--ptxas-options=--warn-on-local-memory-usage",
                    "--ptxas-options=--warn-on-spills",
                    "-gencode",
                    "arch=compute_86,code=sm_86",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
