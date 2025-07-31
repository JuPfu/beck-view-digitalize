from setuptools import setup, Extension
import Cython.Build as cb
import numpy as np
import platform

from glob import glob
from os.path import splitext, basename

compile_args = ["-O3"] if platform.system() != "Windows" else ["/O2"]
pyx_files = glob("*.pyx")

extensions = [
    Extension(
        name=splitext(basename(pyx_file))[0],
        sources=[pyx_file],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args
    )
    for pyx_file in pyx_files
]

setup(
    name='beck-view-digitize',
    version='1.0',
    # packages=['example'],
    # url='',
    license='MIT licence',
    author='juergen pfundt',
    author_email='juergen.pfundt@gmail.com',
    description='cython digitize 16mm films',
    ext_modules = cb.cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "language_level": 3,
        }
    )
)
