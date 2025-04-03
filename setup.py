from setuptools import setup, find_packages

# Define version directly in setup.py
version = "0.2.7"

setup(
    name="NEExT",
    version=version,
    packages=find_packages(),
	include_package_data=True,
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.3,<3.0.0",
        "networkx>=3.0",
        "python-igraph>=0.10.0",
        "pydantic>=2.0.0",
        "scipy",
        "vectorizers==0.2",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",  # For progress bars
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.10.0",
        # Core dependencies for ML functionality
    ],
    extras_require={
        "ml": [
            "xgboost>=1.7.0",
            "imbalanced-learn>=0.10.0",
        ],
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pylint",
            "autopep8",
            "yapf",
            "pre-commit",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",  # Read the Docs theme
            "sphinx-autodoc-typehints>=1.25.0",  # Better type hint support
            "sphinx-copybutton>=0.5.0",  # Add copy button to code blocks
            "myst-parser>=2.0.0",  # Markdown support
            "nbsphinx>=0.9.0",  # Jupyter notebook support
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.3.0",
        ],
        "experiments": [
            "seaborn>=0.12.0",
            "matplotlib>=3.7.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "requests>=2.31.0",  # For downloading data
            "tqdm>=4.65.0",  # For progress bars
            "pandas>=2.0.3",  # For data manipulation
            "numpy>=1.24.0",  # For numerical operations
            "plotly>=5.18.0"  # For interactive visualizations
        ]
    },
	package_data={
        '': ['examples/*.py', 'examples/experiments/*.py', 'examples/experiments/README.md']
    },
    exclude_package_data={
        '': ['examples/*']
    },
    author="Ash Dehghan",
    author_email="ash.dehghan@gmail.com",
    description="NEExT: Network Embedding and Explanation Tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashdehghan/NEExT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 