#!/usr/bin/env python3
"""
Setup script for FinSight Common module.

This module provides shared utilities and components for the FinSight platform,
including logging, caching, LLM integrations, and other common functionality.
"""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Read requirements from main requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            # Filter only the dependencies we need for common module
            lines = f.readlines()
            common_deps = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Include only dependencies that are commonly used
                    if any(
                        dep in line.lower()
                        for dep in [
                            "pydantic",
                            "redis",
                            "aiofiles",
                            "colorama",
                            "openai",
                            "google-generativeai",
                            "langchain",
                            "httpx",
                            "tenacity",
                        ]
                    ):
                        common_deps.append(line)
            return common_deps
    return [
        "pydantic>=2.11.7",
        "pydantic-settings>=2.10.1",
        "redis>=6.2.0",
        "aiofiles>=24.1.0",
        "colorama>=0.4.6",
        "openai>=1.92.2",
        "google-generativeai>=0.8.5",
        "langchain>=0.3.26",
        "langchain-openai>=0.3.26",
        "langchain-google-genai>=2.0.10",
        "httpx>=0.28.1",
        "tenacity>=8.5.0",
    ]


setup(
    name="finsight-common",
    version="0.1.0",
    description="Common utilities and shared components for FinSight platform",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="FinSight Team",
    author_email="team@finsight.com",
    url="https://github.com/finsight/common",
    packages=["common", "common.logger", "common.cache", "common.llm"],
    package_dir={"common": "."},
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=8.4.1",
            "pytest-asyncio>=0.23.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial",
    ],
    include_package_data=True,
    zip_safe=False,
)
