# from distutils.core import setup
from setuptools import setup, Extension
import Cython.Build as cb

extensions = [Extension ('*', sources=['*.pyx'])]

setup(
    name='beck-view-digitize',
    version='1.0',
    # packages=['example'],
    # url='',
    license='MIT licence',
    author='juergen pfundt',
    author_email='juergen.pfundt@gmail.com',
    description='cython test',
    ext_modules = cb.cythonize(extensions)
)
