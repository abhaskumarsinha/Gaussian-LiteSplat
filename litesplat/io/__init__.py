"""
LiteSplat I/O
=============

Data import and conversion utilities for the LiteSplat framework.

Modules
-------
- COLMAP: Convert COLMAP/NeRF-style exports into Gaussian + camera JSON files.
- import_gaussians: Load and optionally subsample Gaussian scenes for rendering.

Example
-------
WIP
"""

from .import_gaussians import import_gaussian_scene
from .COLMAP import convert_colmap_to_gaussians  # Assuming COLMAP.py defines this function

__all__ = [
    "import_gaussian_scene",
    "convert_colmap_to_gaussians",
]

__version__ = "0.1.0"
__author__ = "Abhas Kumar Sinha"
__license__ = "Apache-2.0"
__description__ = (
    "I/O utilities for LiteSplat — including COLMAP conversions and Gaussian scene import."
)
