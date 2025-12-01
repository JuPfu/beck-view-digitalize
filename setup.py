from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess

from glob import glob
from os.path import splitext, basename

pyx_files = glob("*.pyx")

def pkg_config(libname, flag):
    try:
        out = subprocess.check_output(["pkg-config", flag, libname], text=True).strip()
        return out.split()
    except Exception:
        return []

# -------------------------------------------------------------
# libpng detection (macOS Homebrew / Linux pkg-config / Windows)
# -------------------------------------------------------------
include_dirs = []
library_dirs = []
libraries = []

# --- macOS Homebrew ---
if sys.platform == "darwin":
    homebrew = "/opt/homebrew"
    if os.path.exists(homebrew):
        include_dirs.append(f"{homebrew}/include")
        library_dirs.append(f"{homebrew}/lib")
        libraries.extend(["png", "z"])

# --- Linux pkg-config ---
include_dirs += pkg_config("libpng", "--cflags-only-I")
library_dirs += pkg_config("libpng", "--libs-only-L")
libraries += [lib[2:] for lib in pkg_config("libpng", "--libs-only-l")]

# --- fallback ---
if not libraries:
    libraries = ["png", "z"]

# Always add NumPy
include_dirs.append(np.get_include())

extensions = [
    Extension(
        name=splitext(basename(pyx_file))[0],
        sources=[pyx_file],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-march=native",  "-fstrict-aliasing"],
    )
    for pyx_file in pyx_files
]

setup(
    name='beck-view-digitize',
    version='1.2',
    url='https://github.com/JuPfu/beck-view-digitalize',
    license='MIT licence',
    author='juergen pfundt',
    author_email='juergen.pfundt@gmail.com',
    description='cython digitize 16mm films',
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=False,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "nonecheck": False,
            "language_level": 3,
            "infer_types": True
        }
     ),
)
