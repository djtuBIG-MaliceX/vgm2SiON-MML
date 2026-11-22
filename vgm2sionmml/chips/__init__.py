"""Chip-specific register decoders."""

from .opm import extract_ym2151
from .opn import extract_opn
from .opl import extract_opl
from .opll import extract_opll
from .psg import extract_ay8910, extract_sn76489
from .nes import extract_nes
from .scc import extract_scc
from .huc6280 import extract_huc6280
from .vsu import extract_vsu
from .gb import extract_gb
from .k005289 import extract_k005289

__all__ = [
    "extract_ay8910",
    "extract_nes",
    "extract_opl",
    "extract_opll",
    "extract_opn",
    "extract_sn76489",
    "extract_ym2151",
    "extract_scc",
    "extract_huc6280",
    "extract_vsu",
    "extract_gb",
    "extract_k005289",
]
