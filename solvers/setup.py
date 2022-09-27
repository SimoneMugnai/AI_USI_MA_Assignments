from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("AI2022MA/solvers/two_opt_with_candidate.pyx"),
)