"""
validation/_common - shared helpers for stepsic validation scripts.

Re-exports everything from ``validation.py`` so scripts can use either::

    from validation import GrowthData, init_cosmology   # existing
    from _common import GrowthData, init_cosmology      # new style

Both import paths work as long as the parent ``validation/`` directory
is on ``sys.path`` (which every run script already arranges via
``sys.path.insert``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure validation.py is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import (  # noqa: F401, E402
    ArrayF,
    ArrayI,
    ArrayC,
    PLANCK2018,
    GrowthData,
    init_cosmology,
    run_lpt,
    generate_field,
    parse_boxsize,
    load_archive,
    setup_matplotlib,
    histogram,
    z_tag,
    add_cosmology_args,
)

__all__ = [
    "ArrayF",
    "ArrayI",
    "ArrayC",
    "PLANCK2018",
    "GrowthData",
    "init_cosmology",
    "run_lpt",
    "generate_field",
    "parse_boxsize",
    "load_archive",
    "setup_matplotlib",
    "histogram",
    "z_tag",
    "add_cosmology_args",
]
