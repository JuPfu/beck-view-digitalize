# from distutils.core import setup
from setuptools import setup, Extension
import Cython.Build as cb
import numpy as np

extensions = [
    Extension (
        name='*',
        sources=['*.pyx'],
        include_dirs=[np.get_include()],  # Where to find 'numpy/arrayobject.h'
        extra_compile_args=["-O3"],
    )
]

setup(
    name='beck-view-digitize',
    version='1.0',
    # packages=['example'],
    # url='',
    license='MIT licence',
    author='juergen pfundt',
    author_email='juergen.pfundt@gmail.com',
    description='cython digitize super v8 films',
    ext_modules = cb.cythonize(extensions)
)
