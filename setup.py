from setuptools import setup, Extension, dist
import sys

major_python_version, minon_python_version = sys.version_info[:2]

if major_python_version < 3:
    raise Exception("Must be using Python 3.")


build_cython = True
# bootstrap numpy and cython installs
import numpy as np

ext = ".pyx" if build_cython else ".c"

extensions = [
    Extension(
        "intpolynomials.intpolynomials",
        ["lib/intpolynomials/intpolynomials" + ext],
        include_dirs = [np.get_include()]
    )
]

if build_cython:

    from Cython.Build import cythonize
    ext = ".pyx"
    extensions = cythonize(
        extensions,
        compiler_directives = {"language_level" : "3"},
        include_path = [
            "lib/intpolynomials/intpolynomials.pxd"
        ]
    )

setup(
    name = 'intpolynomials',
    version = '0.1',
    description = "Basic operations on polynomials with integer coefficients. Implemented in Cython.",
    long_description = "Basic operations on polynomials with integer coefficients. Implemented in Cython.",
    long_description_content_type="text/plain",

    author = "Michael P. Lane",
    author_email = "mlanetheta@gmail.com",

    package_dir = {"": "lib"},
    packages = [
        "intpolynomials"
    ],
    include_package_data = True,

    test_suite = "tests",

    python_requires = ">=3.5",
    install_requires = [
        'Cython>=0.23',
        'numpy>=1.21.6',
        'mpmath>=1.1.0',
        'xxhash>=3.0.0'
    ],

    zip_safe=False,
    ext_modules = extensions
)