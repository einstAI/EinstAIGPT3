"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()




setup(
    name="torchquad",
    version="0.3.x",
    description="Package providing torch-based numerical integration methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esa/torchquad",
    author="ESA Advanced Concepts Team",
    author_email="pablo.gomez@esa.int",
    install_requires=[
        "loguru>=0.5.3",
        "matplotlib>=3.3.3",
        "scipy>=1.6.0",
        "tqdm>=4.56.1",
        "autoray>=0.2.5",
    ],

    extras_require={
        "dev": [
            "black>=20.8b1",
            "flake8>=3.8.4",
            "flake8-bugbear>=20.1.4",
            "flake8-docstrings>=1.5.0",
            "flake8-import-order>=0.18.1",
            "flake8-rst-docstrings>=0.0.14",
            "flake8-rst>=0.7.1",
            "isort>=5.6.4",
            "mypy>=0.790",
            "pydocstyle>=5.1.1",
            "pytest>=6.2.1",
            "pytest-cov>=2.10.1",
            "pytest-mock>=3.4.0",
            "pytest-xdist>=2.1.0",
            "sphinx>=3.4.3",
            "sphinx-autodoc-typehints>=1.11.1",
            "sphinx-rtd-theme>=0.5.0",
            "sphinxcontrib-bibtex>=2.1.2",
            "sphinxcontrib-napoleon>=0.7",

        ],

    },

    packages=["torchquad"],


    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable

        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",

        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",


        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",


    ]
)
