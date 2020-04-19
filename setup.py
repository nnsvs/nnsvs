# coding: utf-8
from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader
from os.path import exists

version = SourceFileLoader('dnnsvs.version', 'dnnsvs/version.py').load_module().version

packages = find_packages()
if exists("README.md"):
    with open("README.md", "r") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""

setup(name='dnnsvs',
    version=version,
    description='DNN-based singing voice synthesis library',
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    packages=packages,
    include_package_data=True,
    install_requires=[
        "numpy",
        "cython",
        "torch >= 1.1.0",
        "hydra-core",
        "hydra_colorlog",
        "librosa >= 0.7.0",
        "pysptk",
        "pyworld",
        "tensorboard",
    ],
    extras_require={
        "test": [
        ],
    },
    entry_points={
        "console_scripts": [
        ],
    },
    )
