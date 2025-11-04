"""
LiteSplat Core
==============

Core differentiable components of the LiteSplat framework —
a minimal, portable Gaussian Splatting sandbox for small-scale
experimentation and performance insights in pure Python and Keras.

Modules
-------
- CameraLayer: Keras layer for camera projection and Jacobian computation.
- GaussianParameterLayer: Keras layer that manages trainable Gaussian parameters.

Example
-------
WIP
"""

from .camera_model import CameraLayer
from .gaussian_model import GaussianParameterLayer

__all__ = [
    "CameraLayer",
    "GaussianParameterLayer",
]

__version__ = "0.1.0"
__author__ = "Abhas Kumar Sinha"
__license__ = "Apache-2.0"
__description__ = (
    "Core differentiable layers for LiteSplat — "
    "a minimal, portable Gaussian Splatting framework in pure Python and Keras."
)
