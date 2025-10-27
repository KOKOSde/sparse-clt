"""
Setup script for sparse-transcoder package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version
with open("sparse_transcoder/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

setup(
    name="sparse-transcoder",
    version=version,
    author="Fahad Alghanim",
    author_email="fkalghan@email.sc.edu",
    description="Optimized sparse feature extraction for transformer models (LLMs/VLMs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KOKOSde/sparse-transcoder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    keywords=[
        "pytorch",
        "transformers",
        "interpretability",
        "feature-extraction",
        "sparse-coding",
        "mechanistic-interpretability",
        "llm",
        "vision-language-models",
    ],
    project_urls={
        "Bug Reports": "https://github.com/KOKOSde/sparse-transcoder/issues",
        "Source": "https://github.com/KOKOSde/sparse-transcoder",
        "Documentation": "https://github.com/KOKOSde/sparse-transcoder#readme",
    },
)

