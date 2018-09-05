# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:59:29 2017

@author: qcy
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# setup(
#     ext_modules=[
#         Extension("DataAnalyze", ["DataAnalyze.c"],
#                   include_dirs=[np.get_include()]),
#     ],
# )


setup(name = 'ops vis',
      include_dirs=[np.get_include()],
      ext_modules = cythonize("DataAnalyze.pyx"))