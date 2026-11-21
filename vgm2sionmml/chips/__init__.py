"""Chip-specific register decoders."""

from .opm import extract_ym2151
from .opn import extract_opn
from .opl import extract_opl
from .psg import extract_ay8910, extract_sn76489
from .nes import extract_nes

__all__ = [
    "extract_ay8910",
    "extract_nes",
    "extract_opl",
    "extract_opn",
    "extract_sn76489",
    "extract_ym2151",
]
