#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

# setup(
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[Extension("cos_module", ["cos_module1.pyx"])]
# )

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("DataOperator",
                 sources=["_DataOperator.pyx", "DataOperator.c"],
                 include_dirs=[np.get_include()])],
)