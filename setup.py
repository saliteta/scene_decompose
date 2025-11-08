#!/usr/bin/env python3
"""
Root setup script for scene-decompose package
"""

from setuptools import setup, find_packages

# Read the README file
with open("scene_decompose/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scene-decompose",
    version="0.2.0",
    author="Butian Xiong",
    author_email="xiongbutian768@gmail.com",
    description="Hierarchical Gaussian Splatting with Query System for Scene Decomposition, also GT visualization tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scene-decompose",
    packages=find_packages(),
    package_data={
        "scene_decompose": [
            "README.md",
            "*.py",
            "hierachical_gs/*.py",
            "hierachical_gs/splatLayer/*.py",
            "hierachical_gs/tree/*.py", 
            "query_system/*.py",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=1.0.0",
        "tyro>=0.5.0",
        "gsplat-ext",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "visualization": [
            "gradio>=3.0",
            "plotly>=5.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "construct-hierarchical-gs=scene_decompose.construct_hierachical_gs:main",
            "hierarchical-viewer=scene_decompose.hierachical_viewer:main",
        ],
    },
)