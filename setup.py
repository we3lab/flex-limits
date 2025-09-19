from setuptools import setup, find_packages
from pathlib import Path

setup_requirements = []

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text()

test_requirements = [
    "black>=22.3.0",
    "flake8>=4.0.0",
    "codecov>=2.1.4",
    "pytest>=8.1.1",
    "pytest-cov>=3.0.0",
    "pytest-html>=3.1.1",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "Sphinx==7.0.1",
    "sphinx-rtd-theme==2.0.0",
    "tox>=3.24.5",
    "matplotlib>=3.8.4",
    "ipykernel",
    "joblib>=1.3.2",
]

requirements = [
    "pandas>=2.2.1",
    "numpy>=1.26.4",
    "cvxpy>=1.3.0",
    "pyomo>=6.7",
    "gurobipy>=11.0",
    "pint>=0.19.2",
    "matplotlib>=3.8.4",
    'seaborn', 
    'openpyxl', 
    "ipykernel",
    "eeco>=0.1.0",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="WE3Lab",
    author_email="raoak@stanford.edu",
    url="https://github.com/we3lab/flex-limits",
    name="flex_limits",
    version="0.0.1",
    description="Analysis to understand the upper bound of benefits from flexibility for industrial loads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    extras_require=extra_requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="flexibility, limits, industrial loads, optimization",
    python_requires=">=3.9,<=3.13",
)
