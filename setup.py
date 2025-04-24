#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import re

# Read the version from __init__.py
with open(os.path.join("isioshmviewer", "__init__.py"), "r") as f:
    init_contents = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_contents, re.M)
    version = version_match.group(1) if version_match else "0.0.0"

# Read the README for the long description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="isioshmviewer",
    version=version,
    author="J. Taylor",
    author_email="your.email@example.com",  # Replace with your email
    description="A library for interacting with ImageStreamIO shared memory streams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/isioShmViewer",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "coloredlogs",
        "matplotlib",  # If you're using it for visualization
    ],
    extras_require={
        "gui": ["PyQt5"],  # For your GUI components
        "dev": ["pytest", "coverage"],  # For development
    },
    entry_points={
        "console_scripts": [
            "isioshmserver=isioshmviewer.shmViewerServer:main",
            "fakeshmstream=isioshmviewer.fakeShmStream:main",
        ],
    },
)