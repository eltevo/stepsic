#!/usr/bin/env bash
# ==========================================================================
# Generate or load glasses, then measure their power, spacing, density, and residual force.
#
# Usage: bash run.sh [--size=small|medium|large] [standard campaign options]
# GLASS_INPUT_MODE is generate (default) or pre-generated. Pre-generated mode
# requires explicit cubical, spherical, and cylindrical snapshot paths.
# ==========================================================================
set -euo pipefail

CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAMPAIGN_DIR

# Repository configuration is trusted and only selects which internal phase
# runs; each phase resolves and records the typed profile independently.
# shellcheck disable=SC1091
source "${CAMPAIGN_DIR}/config.env"
case "${GLASS_INPUT_MODE}" in
    generate)
        ( source "${CAMPAIGN_DIR}/scripts/generation.inc" )
        ;;
    pre-generated)
        for variable in GLASS_SNAP_CUBICAL GLASS_SNAP_SPHERICAL GLASS_SNAP_CYLINDRICAL; do
            if [[ -z "${!variable:-}" || ! -f "${!variable}" ]]; then
                echo "ERROR: ${variable} must name an existing snapshot in pre-generated mode." >&2
                exit 2
            fi
        done
        repository_root="$(cd "${CAMPAIGN_DIR}/.." && pwd)"
        conda run --no-capture-output -n "${STEPSIC_ENV}" \
            env "PYTHONPATH=${repository_root}/..${PYTHONPATH:+:${PYTHONPATH}}" \
            python "${CAMPAIGN_DIR}/scripts/validate-inputs.py" \
            --cubical "${GLASS_SNAP_CUBICAL}" \
            --spherical "${GLASS_SNAP_SPHERICAL}" \
            --cylindrical "${GLASS_SNAP_CYLINDRICAL}" \
            --box-size "${LBOX}" --radius "${R_3D}" --length "${LZ}"
        ;;
    *)
        echo "ERROR: GLASS_INPUT_MODE must be generate or pre-generated." >&2
        exit 2
        ;;
esac

case "${GLASS_PHASE:-all}" in
    all) ( source "${CAMPAIGN_DIR}/scripts/diagnostics.inc" ) ;;
    generation-only) ;;
    *) echo "ERROR: invalid internal glass phase." >&2; exit 2 ;;
esac
