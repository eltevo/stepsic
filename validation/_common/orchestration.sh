#!/usr/bin/env bash
# Declare campaign steps and decide when cached outputs can be reused.

declare -ag _VLIB_DECLARED_STEPS=()
declare -Ag _VLIB_STEP_INPUTS=()
declare -Ag _VLIB_STEP_IMPLEMENTATIONS=()
declare -ag _VLIB_MANIFEST_CONFIG=()
declare -ag _VLIB_GLOBAL_INPUTS=(
    "${_VLIB_COSMOLOGY_FILE}"
)
declare -ag _VLIB_GLOBAL_IMPLEMENTATIONS=(
    "${_VLIB_COMMON_DIR}/lib.sh"
    "${_VLIB_COMMON_DIR}/runtime.sh"
    "${_VLIB_COMMON_DIR}/cosmology.sh"
    "${_VLIB_COMMON_DIR}/steps.sh"
    "${_VLIB_COMMON_DIR}/stepsic.sh"
    "${_VLIB_COMMON_DIR}/orchestration.sh"
    "${_VLIB_COMMON_DIR}/manifest.py"
    "${_VLIB_COMMON_DIR}/profiles.py"
    "${_VLIB_COMMON_DIR}/artifacts.py"
    "${_VLIB_COMMON_DIR}/cosmology_fields.py"
    "${_VLIB_COMMON_DIR}/evaluation.py"
    "${_VLIB_COMMON_DIR}/pairing.py"
    "${_VLIB_COMMON_DIR}/plotting.py"
    "${_VLIB_COMMON_DIR}/result.py"
    "${_VLIB_COMMON_DIR}/snapshots.py"
)
declare -ag _VLIB_PREVIOUS_OUTPUTS=()
declare -ag _VLIB_CURRENT_OUTPUTS=()
declare -ag _VLIB_CURRENT_INPUTS=()
declare -ag _VLIB_CURRENT_IMPLEMENTATIONS=()
_VLIB_CURRENT_MANIFEST=""
_VLIB_CURRENT_STEP=""
_VLIB_LOCK_FD=""
_VLIB_RECORD_SEPARATOR=$'\034'

vlib::declare_steps() {
    if (( $# == 0 )); then
        echo "ERROR: vlib::declare_steps requires at least one step." >&2
        return 2
    fi
    _VLIB_DECLARED_STEPS=()
    local name existing
    for name in "$@"; do
        if [[ ! "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            echo "ERROR: invalid validation step name: ${name}" >&2
            return 2
        fi
        for existing in "${_VLIB_DECLARED_STEPS[@]+"${_VLIB_DECLARED_STEPS[@]}"}"; do
            if [[ "${existing}" == "${name}" ]]; then
                echo "ERROR: duplicate validation step: ${name}" >&2
                return 2
            fi
        done
        _VLIB_DECLARED_STEPS+=("${name}")
    done
}

vlib::list_steps() {
    local index=1 name
    for name in "${_VLIB_DECLARED_STEPS[@]+"${_VLIB_DECLARED_STEPS[@]}"}"; do
        printf '  %2d. %s\n' "${index}" "${name}"
        (( index++ ))
    done
}

vlib::_manifest_append() {
    local map_name="${1}"
    local step="${2}"
    shift 2
    local value path
    declare -n values="${map_name}"
    value="${values[${step}]:-}"
    for path in "$@"; do
        if [[ "${path}" == *"${_VLIB_RECORD_SEPARATOR}"* ]]; then
            echo "ERROR: manifest paths may not contain control characters." >&2
            return 2
        fi
        [[ -z "${path}" ]] && continue
        value+="${_VLIB_RECORD_SEPARATOR}${path}"
    done
    values["${step}"]="${value}"
}

vlib::manifest_input() {
    local step="${1}"
    shift
    vlib::_manifest_append _VLIB_STEP_INPUTS "${step}" "$@"
}

vlib::manifest_implementation() {
    local step="${1}"
    shift
    vlib::_manifest_append _VLIB_STEP_IMPLEMENTATIONS "${step}" "$@"
}

vlib::manifest_evaluator() {
    vlib::manifest_implementation "${1}" "${2}" \
        "${_VLIB_COMMON_DIR}/evaluation.py" \
        "${_VLIB_COMMON_DIR}/result.py"
}

vlib::manifest_config() {
    local value
    for value in "$@"; do
        if [[ "${value}" == *"${_VLIB_RECORD_SEPARATOR}"* ]]; then
            echo "ERROR: manifest configuration contains a control character." >&2
            return 2
        fi
        _VLIB_MANIFEST_CONFIG+=("${value}")
    done
}

vlib::manifest_init() {
    local campaign_dir="${1}"
    local config_file="${2:-}"
    local key variable
    _VLIB_GLOBAL_IMPLEMENTATIONS+=("${BASH_SOURCE[-1]}")
    if [[ -n "${VLIB_PROFILE_FILE:-}" ]]; then
        _VLIB_GLOBAL_INPUTS+=("${VLIB_PROFILE_FILE}" "${VLIB_RESOLVED_PROFILE}")
        vlib::manifest_config \
            "VALIDATION_SIZE=${VLIB_SIZE}" \
            "VALIDATION_PROFILE_SHA256=${VLIB_PROFILE_SHA256}"
        for key in "${VLIB_PROFILE_KEYS[@]+"${VLIB_PROFILE_KEYS[@]}"}"; do
            vlib::manifest_config "${key}=${!key}"
        done
    else
        vlib::manifest_config "VALIDATION_SIZE=${VLIB_SIZE:-custom}"
    fi
    if [[ -n "${config_file}" ]]; then
        _VLIB_GLOBAL_INPUTS+=("${config_file}")
        while IFS= read -r key; do
            vlib::manifest_config "${key}=${!key-}"
        done < <(
            sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "${config_file}"
        )
    fi
    vlib::manifest_config "COSMOLOGY_NAME=${COSMOLOGY_NAME}"
    for key in "${_VLIB_COSMOLOGY_KEYS[@]}"; do
        variable="COSMO_${key}"
        vlib::manifest_config "${variable}=${!variable}"
    done
}

vlib::_manifest_values() {
    local encoded="${1:-}"
    local output_name="${2}"
    declare -n output="${output_name}"
    output=()
    [[ -z "${encoded}" ]] && return 0
    encoded="${encoded#${_VLIB_RECORD_SEPARATOR}}"
    IFS="${_VLIB_RECORD_SEPARATOR}" read -r -a output <<< "${encoded}"
}

vlib::_manifest_command() {
    local action="${1}"
    local manifest="${2}"
    local step="${3}"
    local -a command=(
        python3 "${_VLIB_COMMON_DIR}/manifest.py"
        "${action}"
        --manifest "${manifest}"
        --step "${step}"
    )
    local value
    for value in "${VLIB_EXTRA_ARGS[@]+"${VLIB_EXTRA_ARGS[@]}"}"; do
        command+=("--argument=${value}")
    done
    for value in "${_VLIB_MANIFEST_CONFIG[@]+"${_VLIB_MANIFEST_CONFIG[@]}"}"; do
        command+=("--configuration=${value}")
    done
    for value in "${_VLIB_CURRENT_INPUTS[@]+"${_VLIB_CURRENT_INPUTS[@]}"}"; do
        command+=("--input=${value}")
    done
    for value in "${_VLIB_CURRENT_IMPLEMENTATIONS[@]+"${_VLIB_CURRENT_IMPLEMENTATIONS[@]}"}"; do
        command+=("--implementation=${value}")
    done
    for value in "${_VLIB_CURRENT_OUTPUTS[@]+"${_VLIB_CURRENT_OUTPUTS[@]}"}"; do
        command+=("--output=${value}")
    done
    "${command[@]}"
}

vlib::_release_step_lock() {
    if [[ -n "${_VLIB_LOCK_FD:-}" ]]; then
        flock -u "${_VLIB_LOCK_FD}"
        exec {_VLIB_LOCK_FD}>&-
        _VLIB_LOCK_FD=""
    fi
}

vlib::step_check() {
    local name="${1}"
    shift
    local -a outputs=()
    local output
    for output in "$@"; do
        [[ -n "${output}" ]] && outputs+=("${output}")
    done

    (( _VLIB_CUR_STEP++ )) || true
    local number="${_VLIB_CUR_STEP}"
    local declared="${_VLIB_DECLARED_STEPS[number - 1]:-}"
    if [[ -z "${declared}" || "${declared}" != "${name}" ]]; then
        echo "ERROR: step ${number} is '${name}', declared '${declared:-<none>}'." >&2
        return 2
    fi

    local -a step_inputs=()
    local -a step_implementations=()
    vlib::_manifest_values "${_VLIB_STEP_INPUTS[${name}]:-}" step_inputs
    vlib::_manifest_values \
        "${_VLIB_STEP_IMPLEMENTATIONS[${name}]:-}" step_implementations
    _VLIB_CURRENT_INPUTS=(
        "${step_inputs[@]+"${step_inputs[@]}"}"
        "${_VLIB_PREVIOUS_OUTPUTS[@]+"${_VLIB_PREVIOUS_OUTPUTS[@]}"}"
        "${_VLIB_GLOBAL_INPUTS[@]+"${_VLIB_GLOBAL_INPUTS[@]}"}"
    )
    _VLIB_CURRENT_IMPLEMENTATIONS=(
        "${step_implementations[@]+"${step_implementations[@]}"}"
        "${_VLIB_GLOBAL_IMPLEMENTATIONS[@]+"${_VLIB_GLOBAL_IMPLEMENTATIONS[@]}"}"
    )
    _VLIB_CURRENT_OUTPUTS=("${outputs[@]+"${outputs[@]}"}")
    _VLIB_PREVIOUS_OUTPUTS+=("${outputs[@]+"${outputs[@]}"}")

    if (( VLIB_LIST_STEPS )); then
        return 1
    fi
    if [[ "${name}" == evaluate* && "${VLIB_EVALUATION}" == "skip" ]]; then
        echo "  [skip] step ${number}: ${name}  (--evaluation=skip)"
        return 1
    fi
    if (( VLIB_PLOT_ONLY )) \
        && [[ "${name}" != *plot* && "${name}" != evaluate* ]]; then
        echo "  [skip] step ${number}: ${name}  (--plot-only)"
        return 1
    fi
    if (( number < VLIB_START_STEP )); then
        echo "  [skip] step ${number}: ${name}  (--step=${VLIB_START_STEP})"
        return 1
    fi

    local force=0 forced
    if (( VLIB_FORCE_ALL )); then
        force=1
    else
        for forced in "${VLIB_FORCE_STEPS[@]+"${VLIB_FORCE_STEPS[@]}"}"; do
            [[ "${forced}" == "${name}" ]] && force=1
        done
    fi

    local manifest_dir="${VLIB_CACHE_DIR}/manifests"
    local lock_dir="${VLIB_CACHE_DIR}/locks"
    mkdir -p "${manifest_dir}" "${lock_dir}"
    _VLIB_CURRENT_MANIFEST="${manifest_dir}/${name}.json"
    _VLIB_CURRENT_STEP="${name}"
    exec {_VLIB_LOCK_FD}>"${lock_dir}/${name}.lock"
    flock -x "${_VLIB_LOCK_FD}"

    if (( ! force )) \
    && (( ${#outputs[@]} > 0 )) \
    && vlib::_manifest_command \
        matches "${_VLIB_CURRENT_MANIFEST}" "${name}"; then
        echo "  [cached] step ${number}: ${name}"
        vlib::_release_step_lock
        if [[ "${name}" == "evaluate" && "${VLIB_EVALUATION}" == "gate" ]]; then
            for output in "${outputs[@]+"${outputs[@]}"}"; do
                if [[ "${output}" == */result.json ]]; then
                    local gate_status=0
                    python3 "${_VLIB_COMMON_DIR}/result.py" gate "${output}" \
                        || gate_status=$?
                    if (( gate_status != 0 )); then
                        exit "${gate_status}"
                    fi
                    return 1
                fi
            done
        fi
        return 1
    fi

    rm -f "${_VLIB_CURRENT_MANIFEST}"
    echo ""
    echo "------------------------------------------------------------------------"
    printf '  Step %d: %s\n' "${number}" "${name}"
    echo "------------------------------------------------------------------------"
    _VLIB_STEP_T0="${SECONDS}"
    return 0
}

vlib::step_done() {
    local name="${1}"
    if [[ "${name}" != "${_VLIB_CURRENT_STEP}" ]]; then
        echo "ERROR: completed step '${name}', expected '${_VLIB_CURRENT_STEP}'." >&2
        vlib::_release_step_lock
        return 2
    fi
    local output
    for output in "${_VLIB_CURRENT_OUTPUTS[@]+"${_VLIB_CURRENT_OUTPUTS[@]}"}"; do
        if [[ ! -e "${output}" ]]; then
            echo "ERROR: step '${name}' did not create ${output}." >&2
            vlib::_release_step_lock
            return 2
        fi
    done
    vlib::_manifest_command write "${_VLIB_CURRENT_MANIFEST}" "${name}"
    local elapsed=$(( SECONDS - _VLIB_STEP_T0 ))
    echo "  -> Step '${name}' complete (${elapsed}s)"
    vlib::_release_step_lock
}
