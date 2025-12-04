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
    """Return pkg-config results as a list of tokens."""
    try:
        out = subprocess.check_output(["pkg-config", flag, libname], text=True).strip()
        return out.split()
    except Exception:
        return []

include_dirs = []
library_dirs = []
libraries = []

# ============================================================
# macOS (Homebrew)
# ============================================================
if sys.platform == "darwin":
    # Detect both Arm64 and Intel Homebrew locations
    brew_candidates = [
        "/opt/homebrew",   # Apple Silicon
        "/usr/local"       # Intel macOS
    ]
    for prefix in brew_candidates:
        if os.path.exists(prefix):
            include_dirs.append(f"{prefix}/include")
            library_dirs.append(f"{prefix}/lib")

    # libpng / zlib usually installed as png / z
    libraries.extend(["png", "z"])


# ============================================================
# Linux (pkg-config)
# ============================================================
if sys.platform.startswith("linux"):
    # Include dirs: convert "-I/usr/include" lists to plain dirs
    cflags = pkg_config("libpng", "--cflags-only-I")
    for flag in cflags:
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])

    # Library dirs: convert "-L/usr/lib"
    ldflags = pkg_config("libpng", "--libs-only-L")
    for flag in ldflags:
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])

    # Libraries: convert "-lpng"
    libs = pkg_config("libpng", "--libs-only-l")
    for flag in libs:
        if flag.startswith("-l"):
            libraries.append(flag[2:])

    # Ensure fallback to zlib
    if "z" not in libraries:
        libraries.append("z")


# ============================================================
# Windows (MSVC / MinGW)
# ============================================================
if sys.platform == "win32":
    # Try common vcpkg install paths
    vcpkg = os.getenv("VCPKG_ROOT")
    if vcpkg:
        triplet = "x64-windows" if sys.maxsize > 2**32 else "x86-windows"
        include_dirs.append(f"{vcpkg}/installed/{triplet}/include")
        library_dirs.append(f"{vcpkg}/installed/{triplet}/lib")

    # Windows libraries are usually named libpng / zlib or png / z
    libraries.extend(["png", "zlib"])


# ============================================================
# Fallback for unknown platforms
# ============================================================
if not libraries:
    libraries = ["png", "z"]


# Always add NumPy include path
include_dirs.append(np.get_include())

extensions = [
    Extension(
        name=splitext(basename(pyx_file))[0],
        sources=[pyx_file],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-fstrict-aliasing"
        ] if sys.platform != "win32" else ["/O2"],
    )
    for pyx_file in pyx_files
]

setup(
    name='beck-view-digitize',
    version='1.2',
    url='https://github.com/JuPfu/beck-view-digitalize',
    license='MIT',
    author='juergen pfundt',
    author_email='juergen.pfundt@gmail.com',
    description='cython digitize 16mm films',
    ext_modules=cythonize(
        extensions,
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
