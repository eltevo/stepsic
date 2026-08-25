#!/usr/bin/env python3
"""Enforce the monofonIC FFT reset ordering used for constraint fields."""

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


def prepare(source: Path) -> bool:
    """Return whether *source* was changed, rejecting unknown layouts."""
    text = source.read_text()
    if any(pattern.search(text) for pattern in _VALID_LAYOUTS):
        return False

    updated, count = _BROKEN_LAYOUT.subn(r"\1this->reset();\n\1\2", text)
    if count != 1:
        raise RuntimeError(
            f"expected one known grid_fft reset-ordering target, found {count}"
        )

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
