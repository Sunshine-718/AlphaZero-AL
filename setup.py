from setuptools import setup, Extension
import platform
import pybind11
import shutil
import os


extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # Windows / MSVC
    extra_compile_args = ["/std:c++20", "/openmp", "/O2", "/utf-8"]
elif platform.system() == "Darwin":
    # macOS - Apple Clang 不支持 -fopenmp，需要 brew install libomp
    homebrew_prefix = os.popen("brew --prefix").read().strip() or "/opt/homebrew"
    libomp_prefix = os.popen("brew --prefix libomp").read().strip() or f"{homebrew_prefix}/opt/libomp"
    extra_compile_args = [
        "-std=c++20", "-O3",
        "-Xpreprocessor", "-fopenmp",
        f"-I{libomp_prefix}/include",
    ]
    extra_link_args = [f"-L{libomp_prefix}/lib", "-lomp"]
else:
    # Linux / GCC
    extra_compile_args = ["-std=c++20", "-fopenmp", "-O3", "-march=native"]
    extra_link_args = ["-fopenmp"]

pybind_includes = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
    "src/cpp",
    "."
]

ext_modules = [
    Extension(
        "mcts_cpp",
        ["src/cpp/mcts_bindings.cpp"],
        include_dirs=pybind_includes,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "env_cpp",
        ["src/cpp/env_bindings.cpp"],
        include_dirs=pybind_includes,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="AlphaZero",
    version="1.0",
    description="AlphaZero: C++ MCTS engine + Connect4 environment",
    ext_modules=ext_modules,
)

# 将编译产物移到 src/ 目录
for name in os.listdir():
    if ("mcts_cpp" in name or "env_cpp" in name) and (".pyd" in name or ".so" in name):
        shutil.move(name, f"./src/{name}")

shutil.rmtree('./build', ignore_errors=True)
