"""VGM/VGZ to SiON MML converter.

The public entry point is :func:`vgm2sionmml.convert.convert_file`; the CLI is
``python -m vgm2sionmml``.
"""

from .convert import ConversionError, ConversionResult, convert_file, convert_vgm
from .vgm import VgmFile, load_vgm

__all__ = [
    "ConversionError",
    "ConversionResult",
    "VgmFile",
    "convert_file",
    "convert_vgm",
    "load_vgm",
]

__version__ = "0.1.0"
