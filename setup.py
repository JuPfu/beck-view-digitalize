from setuptools import setup, Extension
import Cython.Build as cb
import numpy as np
import platform
import subprocess
import sys
from glob import glob
from os.path import splitext, basename

# compile flags
compile_args = ["-O3"] if platform.system() != "Windows" else ["/O2"]
# link_args for additional libs not detected by pkg-config
link_args = ["-ldeflate"]

# find all pyx files
pyx_files = glob("*.pyx")

def pkg_config(pkg):
    """Return dict with include_dirs, library_dirs, libraries, extra_compile_args, extra_link_args"""
    try:
        out = subprocess.check_output(["pkg-config", "--cflags", "--libs", pkg], text=True)
        parts = out.strip().split()
        include_dirs = []
        library_dirs = []
        libraries = []
        extra_compile_args = []
        extra_link_args = []
        for p in parts:
            if p.startswith("-I"):
                include_dirs.append(p[2:])
            elif p.startswith("-L"):
                library_dirs.append(p[2:])
            elif p.startswith("-l"):
                libraries.append(p[2:])
            elif p.startswith("-Wl,") or p.startswith("-pthread"):
                extra_link_args.append(p)
            else:
                extra_compile_args.append(p)
        return {
            "include_dirs": include_dirs,
            "library_dirs": library_dirs,
            "libraries": libraries,
            "extra_compile_args": extra_compile_args,
            "extra_link_args": extra_link_args,
        }
    except subprocess.CalledProcessError:
        print(f"pkg-config failed for {pkg}. Make sure it is installed and pkg-config knows about it.")
        sys.exit(1)
    except FileNotFoundError:
        print("pkg-config not found. Please install pkg-config.")
        sys.exit(1)

# get libpng build info
cfg = pkg_config("libpng")

extensions = [
    Extension(
        name=splitext(basename(pyx_file))[0],
        sources=[pyx_file],
        include_dirs=cfg["include_dirs"] + [np.get_include()],
        library_dirs=cfg["library_dirs"],
        libraries=cfg["libraries"] + ["deflate"],  # add deflate explicitly
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_PROFILE", "0"),
        ],
        extra_compile_args=cfg["extra_compile_args"] + compile_args,
        extra_link_args=cfg["extra_link_args"] + link_args
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
            "language_level": 3,
            "infer_types": True
        }
    )
)
