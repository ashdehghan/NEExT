"""Converters for transforming between different graph formats."""

try:
    from .dgl_converter import DGLConverter, DGLConverterConfig
    
    __all__ = ["DGLConverter", "DGLConverterConfig"]
    DGL_AVAILABLE = True
except ImportError:
    # DGL is optional, so we don't raise an error if it's not installed
    DGL_AVAILABLE = False
    __all__ = []