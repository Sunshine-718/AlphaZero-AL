from setuptools import setup
from Cython.Build import cythonize
import numpy
import shutil
import os

try:
    # 支持多个 cython 模块编译
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
    # 编译后移动 .pyd/.so 文件
    folder = './src/'
    for filename in os.listdir(folder):
        if '.cpp' in filename:
            os.remove(folder + filename)
    for filename in os.listdir():
        if ('env_cython' in filename) and ('.pyd' in filename or '.so' in filename):
            shutil.move(filename, f'./src/environments/Connect4/{filename}')

    shutil.rmtree('./build', ignore_errors=True)
