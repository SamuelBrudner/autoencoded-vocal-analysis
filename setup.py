from __future__ import annotations

import re
from pathlib import Path

import setuptools


def _read_version() -> str:
    init_path = Path(__file__).parent / "ava" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in ava/__init__.py")
    return match.group(1)


long_description = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="AVA: Autoencoded Vocal Analysis",
    version=_read_version(),
    author="Jack Goffinet",
    author_email="jack.goffinet@duke.edu",
    description="Generative modeling of animal vocalizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pearsonlab/autoencoded-vocal-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=1.1,!=1.12.0",
        "numpy",
        "matplotlib",
        "joblib",
        "umap-learn",
        "numba",
        "scikit-learn",
        "scipy",
        "bokeh",
        "h5py",
        "pytest",
        "tqdm",
        "affinewarp @ git+https://github.com/ahwillia/affinewarp.git",
    ],
)
