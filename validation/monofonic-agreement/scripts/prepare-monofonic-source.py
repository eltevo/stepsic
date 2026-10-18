#!/usr/bin/env python3
"""Prepare monofonIC's constraint-field reader for the shared white noise.

Two invariants are enforced in ``grid_fft.cc``:

* the FFT object is reset before its dimensions are set;
* the imported field keeps its amplitudes. monofonIC otherwise divides it by
  its sample standard deviation, but stepsic's saved noise has zero Nyquist
  planes, so that division would raise every other mode by about
  ``1/sqrt(1 - 3/N)`` and the two codes would no longer share the realization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile


_VALID_LAYOUTS = (
    re.compile(
        r"this->reset\(\);\s*\n\s*"
        r"for \(size_t i = 0; i < 3; \+\+i\)\s*\n"
        r"\s*this->n_\[i\] = dimsize\[i\];"
    ),
    re.compile(
        r"this->reset\(\);\s*\n\s*"
        r"this->n_\s*=\s*input_dims;"
    ),
)
_BROKEN_LAYOUT = re.compile(
    r"([ \t]+)(for \(size_t i = 0; i < 3; \+\+i\)\n"
    r"[ \t]+this->n_\[i\] = dimsize\[i\];\n"
    r"[ \t]+this->space_ = rspace_id;\n"
    r"\n)"
    r"[ \t]+this->reset\(\);\n"
)


_RESCALE_LOOP = re.compile(
    r"[ \t]*#pragma omp parallel for reduction\(\+ : sum1, sum2\)\n"
    r"[ \t]*for \(size_t i = 0; i < size\(0\); \+\+i\)\n[ \t]*\{\n"
    r"[ \t]*for \(size_t j = 0; j < size\(1\); \+\+j\)\n[ \t]*\{\n"
    r"[ \t]*for \(size_t k = 0; k < size\(2\); \+\+k\)\n[ \t]*\{\n"
    r"[ \t]*this->relem\(i, j, k\) /= stdw;\n"
    r"[ \t]*\}\n[ \t]*\}\n[ \t]*\}\n"
)


def _order_reset(text: str) -> str:
    if any(pattern.search(text) for pattern in _VALID_LAYOUTS):
        return text
    updated, count = _BROKEN_LAYOUT.subn(r"\1this->reset();\n\1\2", text)
    if count != 1:
        raise RuntimeError(
            f"expected one known grid_fft reset-ordering target, found {count}"
        )
    return updated


def _keep_constraint_amplitude(text: str) -> str:
    if "/= stdw" not in text:
        return text
    updated, count = _RESCALE_LOOP.subn("", text)
    if count != 1 or "/= stdw" in updated:
        raise RuntimeError(
            f"expected one known constraint-field rescale loop, found {count}"
        )
    return updated


def prepare(source: Path) -> bool:
    """Return whether *source* was changed, rejecting unknown layouts."""
    text = source.read_text()
    updated = _keep_constraint_amplitude(_order_reset(text))
    if updated == text:
        return False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.name}.tmp-", dir=source.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source)
        directory_fd = os.open(source.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    changed = prepare(args.source)
    state = "prepared" if changed else "invariant satisfied"
    print(f"source {state}: {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
