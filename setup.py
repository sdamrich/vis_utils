from setuptools import setup



configuration = {
    "name": "vis_utils",
    "version": "0.1",
    "description": "Keops utilities for visualization methods",
    "long_description_content_type": "text/x-rst",
    "keywords": "",
    "license": "MIT",
    "packages": ["vis_utils"],
    "install_requires": [
        "numpy >= 1.17",
        "scikit-learn >= 0.22",
        "scipy >= 1.0",
        "numba >= 0.49",
        "torch >= 1.0",
        "pykeops >= 1.4",
        "matplotlib >= 3.0",
    ],
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
