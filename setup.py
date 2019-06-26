from os import path, listdir

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="inplace-abn",
    version="1.0.0",
    author="Lorenzo Porzi",
    author_email="lorenzo@mapillary.com",
    description="In-Place Activate BatchNorm for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mapillary/inplace_abn",
    packages=[
        "inplace_abn"
    ],
    ext_modules=[
        CUDAExtension(
            name="inplace_abn._backend",
            sources=find_sources("src"),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": []
            },
            include_dirs=["include/"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
