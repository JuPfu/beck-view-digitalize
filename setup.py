from setuptools import setup, Extension
import Cython.Build as cb
import numpy as np
import platform
import subprocess
import sys
import os
from glob import glob
from os.path import splitext, basename


SYSTEM = platform.system()

pyx_files = glob("*.pyx")


# --------------------------------------------------------
# Helper: pkg-config wrapper (macOS + Linux)
# --------------------------------------------------------
def try_pkg_config(pkg):
    try:
        out = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", pkg],
            text=True
        )
        parts = out.strip().split()
        inc, libdirs, libs = [], [], []
        extra_c, extra_l = [], []

        for p in parts:
            if p.startswith("-I"):
                inc.append(p[2:])
            elif p.startswith("-L"):
                libdirs.append(p[2:])
            elif p.startswith("-l"):
                libs.append(p[2:])
            elif p.startswith("-Wl,") or p.startswith("-pthread"):
                extra_l.append(p)
            else:
                extra_c.append(p)

        return {
            "include_dirs": inc,
            "library_dirs": libdirs,
            "libraries": libs,
            "extra_compile_args": extra_c,
            "extra_link_args": extra_l,
        }
    except Exception:
        return None



# ========================================================
# PLATFORM: WINDOWS
# ========================================================
if SYSTEM == "Windows":

    # Paths created by your batch script:
    WIN_PREFIX = os.path.join(os.getcwd(), "build", "install")

    include_dirs = [
        np.get_include(),
        os.path.join(WIN_PREFIX, "include"),
    ]

    library_dirs = [
        os.path.join(WIN_PREFIX, "lib")
    ]

    libraries = [
        "spng",
        "libdeflate",
    ]

    extra_compile_args = ["/O2", "/Ot", "/GL"]
    extra_link_args    = ["/LTCG"]

    define_macros = [
        ("_CRT_SECURE_NO_WARNINGS", None),
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ]


# ========================================================
# PLATFORM: macOS or Linux
# ========================================================
else:
    cfg = try_pkg_config("spng")

    if cfg is None:
        print("⚠ pkg-config failed; applying fallback paths")

        # Linux fallback: typical install locations
        include_dirs = [
            np.get_include(),
            "/usr/include",
            "/usr/local/include",
        ]
        library_dirs = [
            "/usr/lib",
            "/usr/local/lib"
        ]
        libraries = ["spng", "deflate"]  # best guess

        extra_compile_args = ["-O3"]
        extra_link_args    = []
    else:
        include_dirs = cfg["include_dirs"] + [np.get_include()]
        library_dirs = cfg["library_dirs"]
        libraries    = cfg["libraries"]

        # ensure libdeflate is present
        if "deflate" not in libraries and "libdeflate" not in libraries:
            libraries.append("deflate")

        extra_compile_args = cfg["extra_compile_args"] + ["-O3"]
        extra_link_args    = cfg["extra_link_args"]

    # macOS: include Homebrew
    if SYSTEM == "Darwin":
        include_dirs.append("/opt/homebrew/include")
        library_dirs.append("/opt/homebrew/lib")

    define_macros = [
        ("CYTHON_PROFILE", "0"),
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ]



# ========================================================
# EXTENSIONS
# ========================================================
extensions = [
    Extension(
        name=splitext(basename(f))[0],
        sources=[f],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
    for f in pyx_files
]


# ========================================================
# SETUP
# ========================================================
setup(
    name="beck-view-digitize",
    version="1.2",
    description="cython digitize 16mm films",
    url="https://github.com/JuPfu/beck-view-digitalize",
    author="juergen pfundt",
    author_email="juergen.pfundt@gmail.com",
    license="MIT",
    ext_modules=cb.cythonize(
        extensions,
        annotate=False,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "nonecheck": False,
            "infer_types": True,
        },
    ),
)
