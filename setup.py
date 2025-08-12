"""
Setup script for Enhanced Emotion Recognition System
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="enhanced-emotion-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced emotion recognition with hybrid architectures and attention mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emotion_recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "nvidia-ml-py3>=7.352.0",
        ],
        "deployment": [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "gradio>=3.40.0",
            "gunicorn>=21.0.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-recognition=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "emotion recognition",
        "deep learning",
        "computer vision",
        "attention mechanisms",
        "pytorch",
        "facial expression recognition",
        "CNN",
        "EfficientNet",
        "machine learning",
        "artificial intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/emotion_recognition/issues",
        "Source": "https://github.com/yourusername/emotion_recognition",
        "Documentation": "https://github.com/yourusername/emotion_recognition/wiki",
    },
)
