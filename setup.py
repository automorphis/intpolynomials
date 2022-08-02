from setuptools import setup, Extension, dist

build_cython = True

try:
    import numpy as np

except ModuleNotFoundError:

    # bootstrap numpy install
    dist.Distribution().fetch_build_eggs(['oldest_supported_numpy'])
    import numpy as np

if build_cython:

    try:
        from Cython.Build import cythonize

    except ModuleNotFoundError:

        # bootstrap Cython install

        dist.Distribution().fetch_build_eggs(['Cython>=0.25'])

        from Cython.Build import cythonize

ext = ".pyx" if build_cython else ".c"

extensions = [
    Extension(
        "intpolynomials.intpolynomials",
        ["lib/intpolynomials/intpolynomials" + ext],
        include_dirs = [np.get_include()]
    )
]

if build_cython:

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
        'oldest_supported_numpy',
        'mpmath>=1.1.0'
    ],

    zip_safe=False,
    ext_modules = extensions
)