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


# ============================================================
# Windows (MSVC / MinGW) — vcpkg detection
# ============================================================
elif sys.platform.startswith("win"):

    triplet_dynamic = "x64-windows"
    triplet_static = "x64-windows-static"

    possible_roots = [
        os.environ.get("VCPKG_ROOT"),
        os.path.expanduser("~/vcpkg"),
        "C:/vcpkg"
    ]

    vcpkg_root = None
    for root in possible_roots:
        if root and os.path.exists(root):
            vcpkg_root = root
            break

    if vcpkg_root:
        # --- Static linking (preferred) ---
        static_include = os.path.join(vcpkg_root, "installed", triplet_static, "include")
        static_lib = os.path.join(vcpkg_root, "installed", triplet_static, "lib")

        # --- Dynamic linking fallback ---
        dyn_include = os.path.join(vcpkg_root, "installed", triplet_dynamic, "include")
        dyn_lib = os.path.join(vcpkg_root, "installed", triplet_dynamic, "lib")

        # 1) Static linking available?
        png_static = os.path.join(static_lib, "libpng16.lib")
        z_static = os.path.join(static_lib, "zlib.lib")

        if os.path.exists(png_static) and os.path.exists(z_static):
            include_dirs.append(static_include)
            library_dirs.append(static_lib)
            libraries.extend(["libpng16", "zlib"])

            windows_compile_args = ["/O2", "/MT"]
            windows_link_args = [
                "/NODEFAULTLIB:MSVCRT"
            ]

            print(">>> Using static libpng16 + zlib from vcpkg (x64-windows-static)")
        else:
            # 2) Fallback: dynamic
            include_dirs.append(dyn_include)
            library_dirs.append(dyn_lib)
            libraries.extend(["libpng16", "zlib"])

            windows_compile_args = ["/O2", "/MD"]
            windows_link_args = []

            print(">>> Using dynamic libpng16 + zlib (fallback)")
    else:
        # Final fallback
        libraries.extend(["libpng16", "zlib"])
        print(">>> WARNING: VCPKG not found — using fallback DLL names")

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
        extra_link_args = windows_link_args if sys.platform.startswith("win") else extra_link_args,
        extra_compile_args = windows_compile_args if sys.platform.startswith("win") else extra_compile_args,
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
