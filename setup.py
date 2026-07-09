#!/usr/bin/env python3
"""
Setup script for AI-Powered Intrusion Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="ai-ids",
    version="1.0.0",
    description="AI-Powered Intrusion Detection System for real-time network monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/King1kim2kyle3/AI-Intrusion-Detection-System",
    license="MIT",
    
    packages=find_packages(exclude=["tests", "docs"]),
    
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.13.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "pytest>=7.4.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "ids-system=main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: Security",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    keywords="intrusion detection security machine-learning network-monitoring",
    
    project_urls={
        "Bug Reports": "https://github.com/King1kim2kyle3/AI-Intrusion-Detection-System/issues",
        "Documentation": "https://github.com/King1kim2kyle3/AI-Intrusion-Detection-System/docs",
        "Source Code": "https://github.com/King1kim2kyle3/AI-Intrusion-Detection-System",
    },
)
