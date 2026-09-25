"""
setup.py — Build the BBJD eigensolver C extension.

Usage
-----
    python setup.py build_ext --inplace

Requirements
------------
    Python  >= 3.9
    NumPy   >= 1.22
    SciPy   >= 1.9
    BLAS/LAPACK (system or via scipy's bundled openblas)
    A C99 compiler
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension

include_dirs = [
    np.get_include(),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'),
]

libraries = []
library_dirs = []
extra_link_args = []
runtime_library_dirs = []

extra_compile_args_extra = []

# Strategy: use scipy's bundled openblas (32-bit int, includes LAPACK).
# This is the most portable approach since scipy always ships it.
try:
    import scipy
    scipy_libs = os.path.join(os.path.dirname(scipy.__file__), '.libs')
    if not os.path.isdir(scipy_libs):
        # Some installs use scipy.libs at the site-packages level
        scipy_libs = os.path.join(
            os.path.dirname(os.path.dirname(scipy.__file__)), 'scipy.libs')

    if os.path.isdir(scipy_libs):
        # Find the openblas .so
        openblas_so = None
        for f in os.listdir(scipy_libs):
            if 'openblas' in f and f.endswith('.so'):
                openblas_so = f
                break
        if openblas_so:
            # Link directly against the .so
            lib_name = openblas_so.replace('lib', '', 1).rsplit('.so', 1)[0]
            libraries.append(lib_name)
            library_dirs.append(scipy_libs)
            runtime_library_dirs.append(scipy_libs)
            extra_compile_args_extra = ['-DBBJD_SCIPY_OPENBLAS']
            print(f"Using scipy's OpenBLAS: {scipy_libs}/{openblas_so}")
        else:
            raise FileNotFoundError("No openblas .so in scipy.libs")
    else:
        raise FileNotFoundError("scipy.libs not found")
except Exception as e:
    print(f"scipy openblas not found ({e}), trying system BLAS/LAPACK")
    # Fallback: system BLAS/LAPACK via symlinks in our directory
    mydir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(mydir, 'libblas.so')):
        libraries.extend(['blas', 'lapack'])
        library_dirs.append(mydir)
        runtime_library_dirs.append(mydir)
    else:
        # Last resort: hope they're on the default path
        libraries.extend(['lapack', 'blas'])

if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/DNDEBUG', '/openmp']
    extra_link_args.append('/openmp')
else:
    extra_compile_args = [
        '-O3', '-march=native',
        '-fno-math-errno', '-fno-signed-zeros',
        '-DNDEBUG', '-std=c11',
        '-fopenmp',              # for NUMA first-touch and potential
                                 # future parallel kernels
    ] + extra_compile_args_extra
    extra_link_args.append('-fopenmp')

ext = Extension(
    name='_bbjd_c',
    sources=[
        'src/_bbjd_c.c',
        'src/jd_core.c',
    ],
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c',
)

setup(
    name='bbjd',
    version='0.1.0',
    description='Block-Bordered Jacobi-Davidson eigensolver',
    ext_modules=[ext],
    license="GPL-3.0-or-later",
    python_requires='>=3.9',
    install_requires=['numpy>=1.22', 'scipy>=1.9'],
)
