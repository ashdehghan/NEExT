"""Optional local web workbench for NEExT.

This package is intentionally lightweight at import time. Web framework imports
live in ``NEExT.web.app`` and ``NEExT.web.launcher`` so the core NEExT library
does not require the optional ``NEExT[web]`` dependencies.
"""

__all__ = []
