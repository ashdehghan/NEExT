# This file makes the directory a Python package
# Add any imports or code that should be available when someone imports NEExT

# Version of the package
__version__ = "0.2.10"

# Make the class directly available when someone does NEExT()
from .framework import NEExT

# This allows "from NEExT import *"
__all__ = ['NEExT']
