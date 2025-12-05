from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
from glob import glob
from os.path import splitext, basename

#
# -------------------------------------------------------------
# Collect project-local .pyx files (avoid picking up others)
# -------------------------------------------------------------
#
pyx_files = glob("*.pyx")


def pkg_config(libname, flag):
    """Return pkg-config flags split into tokens."""
    try:
        out = subprocess.check_output(["pkg-config", flag, libname],
                                      text=True).strip()
        return out.split()
    except Exception:
        return []


include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []
extra_compile_args = []

#
# ───────────────────────────────────────────────────────────────
#  macOS (Homebrew)
# ───────────────────────────────────────────────────────────────
#
if sys.platform == "darwin":
    brew_prefixes = [
        "/opt/homebrew",   # ARM macOS
        "/usr/local"       # Intel macOS
    ]

    for prefix in brew_prefixes:
        if os.path.exists(prefix):
            include_dirs.append(f"{prefix}/include")
            library_dirs.append(f"{prefix}/lib")

    # macOS always uses dynamic libpng/zlib shipped by Homebrew
    libraries.extend(["png", "z"])

    extra_compile_args = ["-O3", "-march=native", "-fstrict-aliasing"]


#
# ───────────────────────────────────────────────────────────────
#  Linux (pkg-config)
# ───────────────────────────────────────────────────────────────
#
elif sys.platform.startswith("linux"):

    # Extract include dirs (-Ifoo)
    for flag in pkg_config("libpng", "--cflags-only-I"):
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])

    # Extract library dirs (-Lfoo)
    for flag in pkg_config("libpng", "--libs-only-L"):
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])

    # Extract libs (-lpng16)
    for flag in pkg_config("libpng", "--libs-only-l"):
        if flag.startswith("-l"):
            libraries.append(flag[2:])

    if "z" not in libraries:
        libraries.append("z")

    extra_compile_args = ["-O3", "-march=native", "-fstrict-aliasing"]


#
# ───────────────────────────────────────────────────────────────
#  Windows (MSVC + vcpkg) with STATIC linking
# ───────────────────────────────────────────────────────────────
#
elif sys.platform.startswith("win"):

    triplet = "x64-windows"
    vcpkg_roots = [
        os.environ.get("VCPKG_ROOT"),
        "C:/vcpkg",
        os.path.expanduser("~/vcpkg")
    ]

    vcpkg_found = None
    for root in vcpkg_roots:
        if root and os.path.exists(root):
            vcpkg_found = root
            break

    if vcpkg_found:
        include_dirs.append(os.path.join(vcpkg_found, "installed", triplet, "include"))
        library_dirs.append(os.path.join(vcpkg_found, "installed", triplet, "lib"))

        #
        # Static linking: explicitly link against .lib static archives
        #
        static_libpng = os.path.join(vcpkg_found, "installed", triplet, "lib", "libpng16_static.lib")
        static_zlib   = os.path.join(vcpkg_found, "installed", triplet, "lib", "zlib.lib")

        if os.path.exists(static_libpng) and os.path.exists(static_zlib):
            #
            # STATIC linking mode
            #
            extra_link_args = [
                static_libpng,
                static_zlib,
            ]
            libraries = []         # No dll-based libs
        else:
            #
            # Fallback to dynamic linking
            #
            libraries = ["png16", "z"]

        extra_compile_args = ["/O2", "/fp:fast", "/GL"]


#
# ───────────────────────────────────────────────────────────────
#  Fallback (unknown system)
# ───────────────────────────────────────────────────────────────
#
if not libraries and not extra_link_args:
    libraries = ["png", "z"]


#
# Always add NumPy includes
#
include_dirs.append(np.get_include())


#
# ───────────────────────────────────────────────────────────────
#  Build Cython extensions
# ───────────────────────────────────────────────────────────────
#
extensions = [
    Extension(
        name=splitext(basename(pyx))[0],
        sources=[pyx],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,         # empty if statically linked
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
    )
    for pyx in pyx_files
]


#
# ───────────────────────────────────────────────────────────────
#  Final setup()
# ───────────────────────────────────────────────────────────────
#
setup(
    name="beck-view-digitize",
    version="1.2",
    description="cython digitize 16mm films",
    url="https://github.com/JuPfu/beck-view-digitalize",
    author="juergen pfundt",
    license="MIT",
    ext_modules=cythonize(
        extensions,
        annotate=False,
        compiler_directives=dict(
            boundscheck=False,
            wraparound=False,
            initializedcheck=False,
            cdivision=True,
            nonecheck=False,
            language_level=3,
            infer_types=True
        )
    ),
)
