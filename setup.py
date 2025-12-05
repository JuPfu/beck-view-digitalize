from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
from glob import glob
from os.path import splitext, basename, dirname

# Project-local PYX files (avoids picking up unintended files)
pyx_files = glob("*.pyx")


def pkg_config(libname, flag):
    try:
        out = subprocess.check_output(
            ["pkg-config", flag, libname],
            text=True
        ).strip()
        return out.split()
    except Exception:
        return []


# ============================================================
# Build configuration
# ============================================================

include_dirs = []
library_dirs = []
libraries = []

extra_link_args = []
extra_compile_args = []

static_link = True     # <--- static linking enabled


# ============================================================
# macOS (Homebrew)
# ============================================================
if sys.platform == "darwin":
    brew_prefixes = [
        "/opt/homebrew",     # Apple Silicon
        "/usr/local"         # Intel
    ]

    for prefix in brew_prefixes:
        inc = os.path.join(prefix, "include")
        lib = os.path.join(prefix, "lib")
        if os.path.exists(inc):
            include_dirs.append(inc)
        if os.path.exists(lib):
            library_dirs.append(lib)

    # Try static libs first
    png_static = os.path.join(library_dirs[0], "libpng.a")
    z_static = os.path.join(library_dirs[0], "libz.a")

    if static_link and os.path.exists(png_static):
        extra_link_args.extend([png_static])
    else:
        libraries.append("png")

    if static_link and os.path.exists(z_static):
        extra_link_args.extend([z_static])
    else:
        libraries.append("z")

    extra_compile_args = ["-O3", "-fstrict-aliasing"]


# ============================================================
# Linux (pkg-config + static optional)
# ============================================================
elif sys.platform.startswith("linux"):
    for flag in pkg_config("libpng", "--cflags-only-I"):
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])

    for flag in pkg_config("libpng", "--libs-only-L"):
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])

    png_static = "/usr/lib/libpng.a"
    z_static = "/usr/lib/libz.a"

    if static_link and os.path.exists(png_static):
        extra_link_args.append(png_static)
    else:
        libraries.append("png")

    if static_link and os.path.exists(z_static):
        extra_link_args.append(z_static)
    else:
        libraries.append("z")

    extra_compile_args = ["-O3", "-march=native"]


# ============================================================
# Windows (MSVC) + vcpkg static linking
# ============================================================
elif sys.platform.startswith("win"):
    triplet = "x64-windows-static"

    vcpkg_roots = [
        os.environ.get("VCPKG_ROOT"),
        "C:/vcpkg",
        os.path.expanduser("~/vcpkg"),
    ]

    for root in vcpkg_roots:
        if root and os.path.exists(root):
            inc = os.path.join(root, "installed", triplet, "include")
            lib = os.path.join(root, "installed", triplet, "lib")

            include_dirs.append(inc)
            library_dirs.append(lib)

            # STATIC linking: link .lib, not DLLs
            libraries = ["libpng16", "zlib"]

            # Tell MSVC to build static CRT
            extra_link_args.extend([
                "/NODEFAULTLIB:MSVCRT",
                "/DEFAULTLIB:LIBCMT"
            ])

            break

    extra_compile_args = ["/O2"]


# ============================================================
# Fallback (any OS)
# ============================================================
if not libraries and not extra_link_args:
    libraries = ["png", "z"]


# Add NumPy headers
include_dirs.append(np.get_include())


# ============================================================
# Extension definitions
# ============================================================
extensions = [
    Extension(
        name=splitext(basename(pyx))[0],
        sources=[pyx],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ]
    )
    for pyx in pyx_files
]


# ============================================================
# Final setup
# ============================================================
setup(
    name="beck-view-digitize",
    version="1.3",
    description="Cython digitize 16mm films",
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
            "infer_types": True,
        },
    ),
)
