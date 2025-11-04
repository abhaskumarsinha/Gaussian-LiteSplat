"""
LiteSplat Utils
===============

Utility modules and helper layers for LiteSplat.

Modules
-------
- renderer: High-level Keras renderer that combines multiple CameraLayer outputs.

Example
-------
WIP
"""

from .renderer import Renderer

__all__ = [
    "Renderer",
]

__version__ = "0.1.0"
__author__ = "Abhas Kumar Sinha"
__license__ = "Apache-2.0"
__description__ = (
    "Utility layers and rendering helpers for LiteSplat — "
    "including the high-level Keras Renderer for multi-camera aggregation."
)
