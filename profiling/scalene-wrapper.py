#!/usr/bin/env python3
"""
Generic scalene wrapper for stepsic.

scalene's argument parser tends to consume script arguments instead of
forwarding them.  This wrapper reads the config path from the
STEPSIC_CONFIG environment variable and uses runpy to execute stepsic.py
as if invoked directly.

Usage:
    STEPSIC_CONFIG=profiling/configs/profile-small.toml \
    scalene run --outfile profiling/stepsic-scalene-small.json \
        --- profiling/scalene-wrapper.py
    scalene view profiling/stepsic-scalene-small.json
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)  # project root (parent of profiling/)

stepsic = os.path.join(root, "stepsic.py")
config = os.environ.get("STEPSIC_CONFIG")

if not config:
    raise RuntimeError(
        "STEPSIC_CONFIG is not set.\n"
        "Set it to the path of a stepsic TOML config file, e.g.:\n"
        "  STEPSIC_CONFIG=profiling/configs/profile-small.toml scalene run ..."
    )

if not os.path.isabs(config):
    config = os.path.join(root, config)

sys.argv = [stepsic, config]
runpy.run_path(stepsic, run_name="__main__")
