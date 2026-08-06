# ---------------------------------------------------------------------------
# Internal state - do not access directly from drivers
# ---------------------------------------------------------------------------
_VLIB_CUR_STEP=0        # incremented by each vlib::step_check call
_VLIB_STEP_T0=0         # wall-clock second recorded at step start

# ---------------------------------------------------------------------------
# Globals populated by vlib::parse_args
# ---------------------------------------------------------------------------
VLIB_START_STEP=1
VLIB_FORCE_ALL=0
VLIB_FORCE_STEPS=()
VLIB_PLOT_ONLY=0
VLIB_CLEAN=0
VLIB_LIST_STEPS=0
VLIB_CONFIG_FILE=""
VLIB_EXTRA_ARGS=()

# Set by the driver before steps begin:
VLIB_CACHE_DIR=""       # used by vlib::breadcrumb_* and vlib::cache_path

# ============================================================================
# Conda + execution
# ============================================================================

vlib::init_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found in PATH." >&2
        exit 1
    fi
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
}

# Abort if the named conda environment does not exist.
vlib::ensure_env() {
    local env_name="${1}"
    if ! conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
        echo "ERROR: conda environment '${env_name}' does not exist." >&2
        exit 1
    fi
}

# Run an arbitrary command inside a conda environment.
vlib::run_in_env() {
    local env_name="${1}"
    shift
    conda run --no-capture-output -n "${env_name}" "$@"
}

# Run a Python script inside a conda environment.
vlib::run_python() {
    local env_name="${1}"
    local script="${2}"
    shift 2
    local repository_root
    repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    local python_path="${repository_root}"
    if [[ -n "${PYTHONPATH:-}" ]]; then
        python_path+="${python_path:+:}${PYTHONPATH}"
    fi
    conda run --no-capture-output -n "${env_name}" \
        env "PYTHONPATH=${python_path}" python "${script}" "$@"
}

# Run a shell fragment (string) inside a conda environment.
vlib::run_shell_in_env() {
    local env_name="${1}"
    local fragment="${2}"
    conda run --no-capture-output -n "${env_name}" bash -lc "${fragment}"
}

# Read an environment variable from inside a conda environment.
vlib::env_var_in_env() {
    local env_name="${1}"
    local var_name="${2}"
    conda run --no-capture-output -n "${env_name}" \
        bash -lc "printf '%s' \"\${${var_name}:-}\""
}

# ============================================================================
# Argument parsing
# ============================================================================

# Populate VLIB_* globals from "$@".
# Unknown flags are collected into VLIB_EXTRA_ARGS for case-specific parsing.
vlib::parse_args() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --step=*)        VLIB_START_STEP="${arg#--step=}" ;;
            --plot-only)     VLIB_PLOT_ONLY=1 ;;
            --force)         VLIB_FORCE_ALL=1 ;;
            --force-step=*)
                IFS=',' read -ra VLIB_FORCE_STEPS <<< "${arg#--force-step=}"
                ;;
            --clean)         VLIB_CLEAN=1 ;;
            --list-steps)    VLIB_LIST_STEPS=1 ;;
            --config=*)      VLIB_CONFIG_FILE="${arg#--config=}" ;;
            -h|--help)
                vlib::print_help
                exit 0
                ;;
            *)
                VLIB_EXTRA_ARGS+=("${arg}")
                ;;
        esac
    done
}

# Extract and print the script's header comment (lines 2..end-of-header).
# The header ends at the first line matching ^# ={10,} or ^# -{10,}.
# Strips leading "# " or "#".
vlib::print_help() {
    awk '
        !in_header && /^# [=-]{10}/ { in_header=1; next }
        in_header && /^# [=-]{10}/ { exit }
        in_header { sub(/^# ?/, ""); print }
    ' "${BASH_SOURCE[-1]}"
}

# Source a config.env file, then fill unset cosmology values from common defaults.
vlib::source_config() {
    local cfg="${1:-}"
    if [[ -n "${cfg}" ]]; then
        if [[ ! -f "${cfg}" ]]; then
            echo "ERROR: config file not found: ${cfg}" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "${cfg}"
    fi
    vlib::cosmology::load_defaults
}

# Print a sorted list of all step names, one per line.
vlib::report_done() {
    echo ""
    echo "========================== Pipeline complete =========================="
    echo ""
}

vlib::atomic_text() {
    local target="${1}"
    local directory temporary status=0
    directory="$(dirname "${target}")"
    mkdir -p "${directory}"
    temporary="$(mktemp "${directory}/.${target##*/}.tmp-XXXXXX")"
    cat > "${temporary}" || status=$?
    if (( status == 0 )); then
        sync -f "${temporary}" || status=$?
    fi
    if (( status == 0 )); then
        mv -f "${temporary}" "${target}" || status=$?
    fi
    if (( status == 0 )); then
        sync -f "${target}" || status=$?
    fi
    if [[ -e "${temporary}" ]]; then
        rm -f "${temporary}"
    fi
    return "${status}"
}

# ============================================================================
# Breadcrumb state (inter-step key/value store)
# ============================================================================
#
# Stores persistent key=value entries in ${VLIB_CACHE_DIR}/state.env.
# Subsequent steps (or re-runs starting at --step=N) can retrieve values
# written by earlier steps.
#
# VLIB_CACHE_DIR must be set before calling these functions.

vlib::breadcrumb_set() {
    local key="${1}"
    local value="${2}"
    if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "ERROR: invalid breadcrumb key: ${key}" >&2
        return 2
    fi
    if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
        echo "ERROR: breadcrumb values must be single-line text." >&2
        return 2
    fi

    local state_file="${VLIB_CACHE_DIR}/state.env"
    local temporary lock_fd
    mkdir -p "${VLIB_CACHE_DIR}"
    exec {lock_fd}>"${VLIB_CACHE_DIR}/state.lock"
    flock -x "${lock_fd}"
    temporary="$(mktemp "${VLIB_CACHE_DIR}/.state.env.tmp-XXXXXX")"
    if [[ -f "${state_file}" ]]; then
        grep -v "^${key}=" "${state_file}" > "${temporary}" || true
    fi
    printf '%s=%s\n' "${key}" "${value}" >> "${temporary}"
    sync -f "${temporary}"
    mv -f "${temporary}" "${state_file}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

vlib::breadcrumb_get() {
    local key="${1}"
    local state_file="${VLIB_CACHE_DIR}/state.env"
    [[ ! -f "${state_file}" ]] && { echo ""; return 0; }
    local line
    line="$(grep "^${key}=" "${state_file}" 2>/dev/null | tail -1)"
    echo "${line#*=}"
}

# Return an absolute path inside VLIB_CACHE_DIR.
vlib::cache_path() {
    echo "${VLIB_CACHE_DIR}/${1}"
}

# ============================================================================
# File discovery
# ============================================================================

# Find the IC file (ic.hdf5) beneath a directory; exits on failure.
vlib::find_ic() {
    local directory="${1}"
    local -a matches=()
    mapfile -d '' matches < <(find "${directory}" -name 'ic.hdf5' -print0)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "ERROR: expected one ic.hdf5 under ${directory}; found ${#matches[@]}." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

# Find the most recently modified file matching a glob pattern in a directory.
# Default pattern is StePS' particle-snapshot naming (snapshot_NNNN.hdf5) so
# this never picks up StePS' Ewald lookup tables (Ewald_table_*.hdf5,
# S1R2_Ewald_table_*.hdf5) from OUT_DIR.
# Prints an empty string (no error) if nothing matches.
vlib::find_last_snap() {
    local dir="${1}"
    local pattern="${2:-snapshot_*.hdf5}"
    shopt -s nullglob
    local files=("${dir}"/${pattern})
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        printf ''
        return 0
    fi
    ls -1t "${files[@]}" | head -1
}

# ============================================================================
# Directory cleanup
# ============================================================================

# Remove all contents of a directory (create it first if needed).
vlib::clear_dir() {
    local dir="${1}"
    mkdir -p "${dir}"
    find "${dir}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

# Remove files matching glob patterns inside a directory.
# Usage: vlib::clear_files <dir> <pattern1> [<pattern2> ...]
vlib::clear_files() {
    local dir="${1}"
    shift
    mkdir -p "${dir}"
    shopt -s nullglob
    local pattern
    for pattern in "$@"; do
        local matches=("${dir}"/${pattern})
        if [[ ${#matches[@]} -gt 0 ]]; then
            rm -f "${matches[@]}"
        fi
    done
    shopt -u nullglob
}
