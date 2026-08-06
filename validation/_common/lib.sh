#!/usr/bin/env bash
# Source facade for the validation shell modules.

if [[ -n "${_VLIB_LOADED:-}" ]]; then
    return 0
fi
_VLIB_LOADED=1

_VLIB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=validation/_common/runtime.sh
source "${_VLIB_COMMON_DIR}/runtime.sh"
# shellcheck source=validation/_common/cosmology.sh
source "${_VLIB_COMMON_DIR}/cosmology.sh"
# shellcheck source=validation/_common/steps.sh
source "${_VLIB_COMMON_DIR}/steps.sh"
# shellcheck source=validation/_common/stepsic.sh
source "${_VLIB_COMMON_DIR}/stepsic.sh"
# shellcheck source=validation/_common/orchestration.sh
source "${_VLIB_COMMON_DIR}/orchestration.sh"
