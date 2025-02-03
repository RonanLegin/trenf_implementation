from setuptools import setup, find_packages

setup(
    name="trenf",
    version="0.1.0",
    description="Implementation of Translation and Rotation Equivariant Normalizing Flow (TRENF) for Optimal Cosmological Analysis",
    author="Ronan Legin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)