from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
from glob import glob
from pathlib import Path


# ================================================================
# 1) Find .pyx sources recursively
# ================================================================
pyx_files = glob("*.pyx")

if not pyx_files:
    print("WARNING: No .pyx files found! (search path: **/*.pyx)")
else:
    print(f"Found {len(pyx_files)} .pyx files")


# ================================================================
# 2) Helper: safe pkg-config invocation (Linux/macOS)
# ================================================================
def pkg_config_flags(libname, flag):
    try:
        out = subprocess.check_output(
            ["pkg-config", flag, libname],
            text=True
        ).strip()
        return out.split()
    except Exception:
        return []


# ================================================================
# 3) Platform configuration
# ================================================================
include_dirs = []
library_dirs = []
libraries = []

PLATFORM = sys.platform


# ================================================================
# macOS (Homebrew)
# ================================================================
if PLATFORM == "darwin":
    print("Configuring for macOS…")

    brew_paths = [
        "/opt/homebrew",  # Apple Silicon
        "/usr/local"      # Intel macOS
    ]

    for prefix in brew_paths:
        if os.path.exists(prefix):
            include_dirs.append(f"{prefix}/include")
            library_dirs.append(f"{prefix}/lib")
            print(f"Using Homebrew libs from {prefix}")
            break

    libraries += ["png", "z"]  # macOS names


# ================================================================
# Linux (pkg-config)
# ================================================================
elif PLATFORM.startswith("linux"):
    print("Configuring for Linux…")

    # --- Includes ---
    for flag in pkg_config_flags("libpng", "--cflags-only-I"):
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])

    # --- Lib dirs ---
    for flag in pkg_config_flags("libpng", "--libs-only-L"):
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])

    # --- Libraries ---
    for flag in pkg_config_flags("libpng", "--libs-only-l"):
        if flag.startswith("-l"):
            libraries.append(flag[2:])

    if "z" not in libraries:
        libraries.append("z")


# ================================================================
# Windows (MSVC / vcpkg)
# ================================================================
elif PLATFORM.startswith("win"):
    print("Configuring for Windows…")

    # Known places for vcpkg
    vcpkg_roots = [
        os.environ.get("VCPKG_ROOT"),
        str(Path.home() / "vcpkg"),
        "C:/vcpkg"
    ]

    found_vcpkg = False
    for root in vcpkg_roots:
        if root and os.path.exists(root):
            triplet = "x64-windows"
            inc = Path(root) / "installed" / triplet / "include"
            lib = Path(root) / "installed" / triplet / "lib"

            if inc.exists():
                include_dirs.append(str(inc))
                library_dirs.append(str(lib))
                libraries += ["libpng16", "zlib"]  # vcpkg names
                print(f"Using vcpkg libs from {root}")
                found_vcpkg = True
                break

    if not found_vcpkg:
        print("WARNING: vcpkg not found → PNG support disabled")
        libraries = []  # PNG is optional on Windows


# ================================================================
# Fallback
# ================================================================
if not libraries:
    print("NOTE: PNG libraries missing, using fallback (no PNG linking)")
    libraries = []


# Always add NumPy include directory
include_dirs.append(np.get_include())


# ================================================================
# 4) Build extension list
# ================================================================
extensions = [
    Extension(
        name=str(Path(pyx).with_suffix("")).replace(os.sep, "."),
        sources=[pyx],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],
        extra_compile_args=(
            ["/O2"] if PLATFORM.startswith("win") else ["-O3", "-march=native", "-fstrict-aliasing"]
        )
    )
    for pyx in pyx_files
]


# ================================================================
# 5) setup()
# ================================================================
setup(
    name="beck-view-digitize",
    version="1.3",
    description="Cython accelerated 16mm digitization tools",
    author="juergen pfundt",
    url="https://github.com/JuPfu/beck-view-digitalize",
    ext_modules=cythonize(
        extensions,
        annotate=False,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "language_level": 3,
            "infer_types": True,
        },
    ),
)
