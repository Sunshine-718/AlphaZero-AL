from setuptools import setup, Extension
from Cython.Build import cythonize
import platform
import pybind11
import shutil
import numpy
import os


extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # Windows / MSVC
    extra_compile_args = ["/std:c++20", "/openmp", "/O2", "/utf-8"]
else:
    # Linux / GCC / Clang
    extra_compile_args = ["-std=c++20", "-fopenmp", "-O3", "-march=native"]
    extra_link_args = ["-fopenmp"]

try:

    ext_modules = [
        Extension(
            "mcts_cpp",  # 生成的包名
            ["src/bindings.cpp"],  # 只需要编译这个绑定文件，它会 include 头文件
            include_dirs=[
                pybind11.get_include(),  # pybind11 头文件路径
                pybind11.get_include(user=True),
                "src",  # 你的 .h 文件所在的路径
                "."
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
    setup(
        name="AlphaZeroMCTS",
        version="1.0",
        description="High performance MCTS with C++20 and pybind11",
        ext_modules=ext_modules,
    )
    
    ext_modules = cythonize(
        ["./src/env_cython.pyx"],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False
        }
    )
    setup(
        ext_modules=ext_modules,
        include_dirs=[numpy.get_include()],
    )
except Exception as e:
    print(f"[Error] Cython 编译失败: {e}")

finally:
    for name in os.listdir():
        if "mcts_cpp" in name:
            shutil.move(name, f"./src/{name}")
    folder = './src/'
    for filename in os.listdir(folder):
        if '.cpp' in filename and filename != "bindings.cpp":
            os.remove(folder + filename)
    for filename in os.listdir():
        if ('env_cython' in filename) and ('.pyd' in filename or '.so' in filename):
            shutil.move(filename, f'./src/environments/Connect4/{filename}')

    shutil.rmtree('./build', ignore_errors=True)
