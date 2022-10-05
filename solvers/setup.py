from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("two_opt_with_candidate.pyx"),
)