#!/usr/bin/env bash
# ============================================================================
#  validation/_common/lib.sh - shared bash library for stepsic validation
#
#  Source this at the top of every driver:
#    source "$(dirname "$0")/../_common/lib.sh"
#
#  Provides:
#    - Conda environment management  (vlib::init_conda, vlib::ensure_env, …)
#    - Argument parsing              (vlib::parse_args, vlib::print_help)
#    - Step-gated pipeline control   (vlib::step_check, vlib::step_done)
#    - Breadcrumb inter-step state   (vlib::breadcrumb_set/get)
#    - File discovery utilities      (vlib::find_ic, vlib::find_last_snap)
#    - Directory cleanup             (vlib::clear_dir, vlib::clear_files)
#    - Cosmology helpers             (vlib::cosmology::load, ::eds_h0)
#    - StePS build / run helpers     (vlib::steps::*)
#    - Stepsic TOML generator        (vlib::stepsic::write_toml)
#
#  Standard driver skeleton:
#    source "$(dirname "$0")/../_common/lib.sh"
#    vlib::parse_args "$@"
#    vlib::source_config "${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
#    vlib::init_conda
#    vlib::ensure_env "${STEPSIC_ENV:-stepsic}"
#    VLIB_CACHE_DIR="${BASEDIR}/cache"
#    mkdir -p "${VLIB_CACHE_DIR}" "${BASEDIR}/output"
#    if vlib::step_check "run" "${VLIB_CACHE_DIR}/data.npz"; then
#        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" …
#        vlib::step_done "run"
#    fi
#    vlib::report_done
# ============================================================================

# Double-source guard.
if [[ -n "${_VLIB_LOADED:-}" ]]; then
    return 0
fi
_VLIB_LOADED=1

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
    conda run --no-capture-output -n "${env_name}" python "${script}" "$@"
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
    sed -n '2,/^# [=\-]\{10,\}/{/^# [=\-]\{10,\}/d; s/^# \?//; p;}' \
        "${BASH_SOURCE[-1]}"
}

# Source a config.env file; no-op if path is empty.
vlib::source_config() {
    local cfg="${1:-}"
    [[ -z "${cfg}" ]] && return 0
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: config file not found: ${cfg}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${cfg}"
}

# Print a sorted list of all step names, one per line.
vlib::list_steps() {
    local i=1
    local name
    for name in "${_VLIB_STEP_NAMES[@]+"${_VLIB_STEP_NAMES[@]}"}"; do
        printf '  %2d. %s\n' "${i}" "${name}"
        (( i++ ))
    done
}
_VLIB_STEP_NAMES=()  # filled by vlib::step_check

# ============================================================================
# Step-gated pipeline control
# ============================================================================
#
# Usage pattern (in a driver):
#
#   if vlib::step_check "run" "${CACHE}/data.npz"; then
#       vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" …
#       vlib::step_done "run"
#   fi
#
# vlib::step_check <name> [<output_path> ...]
#   Increments the internal step counter.
#   Returns 0 (caller should run the step body) or 1 (skip).
#   Handles: --list-steps, --plot-only, --step=N, --force, --force-step=…,
#            and existence-based cache skipping.
#
vlib::step_check() {
    local name="${1}"
    shift
    local outputs=("$@")

    _VLIB_STEP_NAMES+=("${name}")
    (( _VLIB_CUR_STEP++ )) || true
    local n="${_VLIB_CUR_STEP}"

    # --list-steps: enumerate without running
    if (( VLIB_LIST_STEPS )); then
        printf '  %2d. %s\n' "${n}" "${name}"
        return 1
    fi

    # --plot-only: skip steps whose name does not contain "plot"
    if (( VLIB_PLOT_ONLY )) && [[ "${name}" != *plot* ]]; then
        echo "  [skip] step ${n}: ${name}  (--plot-only)"
        return 1
    fi

    # --step=N: skip steps before N
    if (( n < VLIB_START_STEP )); then
        echo "  [skip] step ${n}: ${name}  (--step=${VLIB_START_STEP})"
        return 1
    fi

    # Determine whether force is requested for this specific step
    local force=0
    if (( VLIB_FORCE_ALL )); then
        force=1
    else
        local f
        for f in "${VLIB_FORCE_STEPS[@]+"${VLIB_FORCE_STEPS[@]}"}"; do
            if [[ "${f}" == "${name}" ]]; then
                force=1
                break
            fi
        done
    fi

    if (( force )); then
        local o
        for o in "${outputs[@]+"${outputs[@]}"}"; do
            [[ -e "${o}" ]] && rm -rf "${o}"
        done
    else
        # Cache check: skip only when ALL declared outputs exist
        if [[ ${#outputs[@]} -gt 0 ]]; then
            local all_exist=1
            local o
            for o in "${outputs[@]}"; do
                if [[ ! -e "${o}" ]]; then
                    all_exist=0
                    break
                fi
            done
            if (( all_exist )); then
                echo "  [cached] step ${n}: ${name}"
                return 1
            fi
        fi
    fi

    echo ""
    echo "------------------------------------------------------------------------"
    printf '  Step %d: %s\n' "${n}" "${name}"
    echo "------------------------------------------------------------------------"
    _VLIB_STEP_T0="${SECONDS}"
    return 0
}

# Log step completion and elapsed time. Call immediately after the step body.
vlib::step_done() {
    local name="${1}"
    local elapsed=$(( SECONDS - _VLIB_STEP_T0 ))
    echo "  -> Step '${name}' complete (${elapsed}s)"
}

# Print a completion banner at the end of the driver.
vlib::report_done() {
    echo ""
    echo "========================== Pipeline complete =========================="
    echo ""
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
    local state_file="${VLIB_CACHE_DIR}/state.env"
    mkdir -p "${VLIB_CACHE_DIR}"
    # Remove any existing entry for this key, then append the new one.
    if [[ -f "${state_file}" ]]; then
        grep -v "^${key}=" "${state_file}" > "${state_file}.tmp" 2>/dev/null || true
        mv "${state_file}.tmp" "${state_file}"
    fi
    printf '%s=%s\n' "${key}" "${value}" >> "${state_file}"
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
    local dir="${1}"
    local result
    result="$(find "${dir}" -name 'ic.hdf5' -print -quit)"
    if [[ -z "${result}" ]]; then
        echo "ERROR: No ic.hdf5 found under ${dir}" >&2
        exit 1
    fi
    echo "${result}"
}

# Find the most recently modified file matching a glob pattern in a directory.
# Default pattern is StePS' particle-snapshot naming (snapshot_NNNN.hdf5) so
# this never picks up Ewald-table caches (Ewald_table_*.hdf5,
# S1R2_Ewald_table_*.hdf5) that StePS also drops into OUT_DIR.
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

# ============================================================================
# Cosmology helpers
# ============================================================================

# Load a cosmology from _common/cosmology/<name>.toml and export COSMO_* vars.
# Parses simple "KEY = VALUE" lines; strips inline comments and quotes.
# Usage: vlib::cosmology::load Planck2018
vlib::cosmology::load() {
    local name="${1}"
    local _lib_dir
    _lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local toml_file="${_lib_dir}/cosmology/${name}.toml"
    if [[ ! -f "${toml_file}" ]]; then
        echo "ERROR: Cosmology file not found: ${toml_file}" >&2
        exit 1
    fi
    while IFS= read -r line; do
        line="${line%%#*}"               # strip inline comment
        [[ -z "${line// }" ]] && continue  # skip blank lines
        local key val
        key="$(awk -F'=' '{gsub(/[[:space:]]/, "", $1); print toupper($1)}' <<< "${line}")"
        val="$(awk -F'=' '{sub(/^[^=]*=/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print}' <<< "${line}")"
        val="${val#\"}"
        val="${val%\"}"
        export "COSMO_${key}=${val}"
    done < "${toml_file}"
    export COSMOLOGY_NAME="${name}"
}

# EdS H0 for the glass-making StePS run.
#
# Always returns 100 km/s/Mpc (h_EdS = 1), independently of the target
# LCDM cosmology. Rationale: stepsic ICs use an h-independent RHO_CRIT
# (canonical H0 = 100 km/s/Mpc), so for StePS's rho_part / rho_cosm
# check to balance, the glass-run HubbleConstant must also use h = 1.
# The physically-motivated choice H0_EdS = H0_LCDM * sqrt(Omega_m_LCDM)
# (which preserves the absolute matter density in physical units) leaves
# h_EdS != 1, which exposes a unit mismatch in StePS's PERIODIC_Z mass
# check (Rsim is not converted by H_INDEPENDENT_UNITS=1 while L is) and
# trips the consistency assertion with rho_part / rho_cosm = 1 / h_EdS^2.
# H0 affects only the simulation timescale; comoving relaxation is
# invariant, so picking h = 1 is safe for glass-making.
#
# Usage: vlib::cosmology::eds_h0 <H0> <Omega_m>   (args ignored)
vlib::cosmology::eds_h0() {
    echo "100.0"
}

# ============================================================================
# StePS build and run helpers
# ============================================================================
#
# Required env vars (set in config.env or driver):
#   STEPS_SRC     - path to StePS source tree
#   STEPS_ENV     - conda env name for StePS
#   BUILD_DIR     - where compiled binaries are installed
#   STEPS_BACKEND - cuda (default) or bh
#   N_MPI, N_GPU, OMP_NUM_THREADS

vlib::steps::validate_backend() {
    case "${STEPS_BACKEND:-cuda}" in
        cuda|bh) ;;
        *)
            echo "ERROR: STEPS_BACKEND must be 'cuda' or 'bh' (got '${STEPS_BACKEND}')." >&2
            return 2
            ;;
    esac
}

# Detect compiler/library paths from the StePS conda env.
# Idempotent: only runs once per shell session.
# Sets: CXX, CUDA_PATH, MPI_INC, MPI_LIBS, HDF5_INC, HDF5_LIBS
vlib::steps::detect_toolchain() {
    [[ -n "${_VLIB_TOOLCHAIN_DETECTED:-}" ]] && return 0
    _VLIB_TOOLCHAIN_DETECTED=1

    local conda_prefix
    conda_prefix="$(vlib::env_var_in_env "${STEPS_ENV}" CONDA_PREFIX)"
    if [[ -z "${conda_prefix}" ]]; then
        echo "ERROR: Could not determine CONDA_PREFIX inside env '${STEPS_ENV}'." >&2
        exit 1
    fi

    CXX="${CXX:-x86_64-conda-linux-gnu-c++}"

    # Resolve CUDA_PATH from the host CUDA toolkit, not the conda env.
    # Priority: (1) already set, (2) nvcc in PATH, (3) /usr/local/cuda,
    #           (4) conda prefix (fallback, unlikely to have nvcc).
    if [[ -z "${CUDA_PATH:-}" ]]; then
        local _nvcc_path
        if _nvcc_path="$(command -v nvcc 2>/dev/null)"; then
            CUDA_PATH="$(dirname "$(dirname "${_nvcc_path}")")"
        elif [[ -d "/usr/local/cuda" ]]; then
            CUDA_PATH="/usr/local/cuda"
        else
            CUDA_PATH="${conda_prefix}"
        fi
    fi

    MPI_INC="${MPI_INC:--I${conda_prefix}/include}"

    if [[ -f "${conda_prefix}/lib/libmpi_cxx.so" ]] \
    || [[ -f "${conda_prefix}/lib/libmpi_cxx.a" ]]; then
        MPI_LIBS="${MPI_LIBS:--L${conda_prefix}/lib -lmpi_cxx -lmpi}"
    else
        MPI_LIBS="${MPI_LIBS:--L${conda_prefix}/lib -lmpi}"
    fi

    HDF5_INC="${HDF5_INC:--I${conda_prefix}/include}"
    HDF5_LIBS="${HDF5_LIBS:--L${conda_prefix}/lib -lhdf5}"
}

# StePS force-calculation precision, selected via the STEPS_PRECISION env var.
vlib::steps::precision_flags() {
    case "${STEPS_PRECISION:-double}" in
        double) ;;                                # StePS default: no flag
        single) printf 'USE_SINGLE_PRECISION' ;;
        *)
            echo "ERROR: STEPS_PRECISION must be 'single' or 'double' (got '${STEPS_PRECISION}')." >&2
            exit 1
            ;;
    esac
}

# Binary-name suffix matching the selected precision
vlib::steps::precision_suffix() {
    case "${STEPS_PRECISION:-double}" in
        single) printf '_single' ;;
        *) ;;
    esac
}

# Compile a StePS binary with the given compile-time feature flags.
# Usage: vlib::steps::build <binary_name> <FLAG1> [FLAG2 ...]
# The binary is installed to ${BUILD_DIR}/<binary_name>.
# Flags must exist in the Makefile template; unknown flags are hard errors
# Requires: vlib::steps::detect_toolchain already called; STEPS_SRC, BUILD_DIR set.
vlib::steps::build() {
    local binary_name="${1}"
    shift
    local flags=("$@")
    local build_product using_cuda

    vlib::steps::validate_backend || return
    case "${STEPS_BACKEND:-cuda}" in
        cuda)
            build_product="StePS_CUDA"
            using_cuda="YES"
            ;;
        bh)
            build_product="StePS"
            using_cuda="NO"
            ;;
    esac

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Building ${binary_name}  (backend=${STEPS_BACKEND:-cuda}, flags: ${flags[*]:-none})"
    echo "----------------------------------------------------------------"

    mkdir -p "${BUILD_DIR}"

    local script
    script="$(cat <<EOF
set -euo pipefail
cd "${STEPS_SRC}/StePS"
cp Template-LinuxGCC-Makefile Makefile
EOF
)"

    local flag
    for flag in "${flags[@]+"${flags[@]}"}"; do
        script+=$'\n'
        script+="if ! grep -qE '^#?OPT \\+= -D${flag}\\b' Makefile; then"
        script+=$'\n'
        script+="    echo 'ERROR: Unknown StePS build flag -D${flag} (absent from Makefile template).' >&2"
        script+=$'\n'
        script+="    exit 1"
        script+=$'\n'
        script+="fi"
        script+=$'\n'
        script+="sed -i 's|^#\\(OPT += -D${flag}\\b\\)|\\1|' Makefile"
    done

    script+="$(cat <<EOF

rm -rf build/
make -r \\
    "build/${build_product}" \\
    "USING_CUDA=${using_cuda}" \\
    "CXX=${CXX}" \\
    "CUDA_PATH=${CUDA_PATH}" \\
    "MPI_INC=${MPI_INC}" \\
    "MPI_LIBS=${MPI_LIBS}" \\
    "HDF5_INC=${HDF5_INC}" \\
    "HDF5_LIBS=${HDF5_LIBS}" \\
    "CUDAFLAGS=-Xcompiler -fopenmp -lineinfo --std=c++17 -Xcompiler -Wall -O3 -Xcompiler -pthread"

if [[ ! -f "build/${build_product}" ]]; then
    echo "ERROR: Build failed - build/${build_product} not found." >&2
    exit 1
fi
mkdir -p "${BUILD_DIR}"
cp "build/${build_product}" "${BUILD_DIR}/${binary_name}"
echo "  -> Installed: ${BUILD_DIR}/${binary_name}"
EOF
)"

    vlib::run_shell_in_env "${STEPS_ENV}" "${script}"
}

# Run a StePS binary with MPI + backend-specific worker dispatch inside the StePS conda env.
# Usage: vlib::steps::run_binary <binary_path> <param_path>
vlib::steps::run_binary() {
    local binary="${1}"
    local param="${2}"
    local parallel_count
    local script

    vlib::steps::validate_backend || return
    case "${STEPS_BACKEND:-cuda}" in
        cuda) parallel_count="${N_GPU}" ;;
        bh)   parallel_count="${OMP_NUM_THREADS}" ;;
    esac

    script="$(cat <<EOF
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
mpirun -np "${N_MPI}" "${binary}" "${param}" "${parallel_count}"
EOF
)"
    vlib::run_shell_in_env "${STEPS_ENV}" "${script}"
}
# Run the validation-owned Ewald cache helper inside the stepsic environment.
# Arguments are passed structurally; no caller-provided path is evaluated as shell code.
vlib::steps::_ewald_cache_helper() {
    local helper_dir
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    vlib::run_in_env "${STEPSIC_ENV}" python \
        "${helper_dir}/ewald_cache.py" "$@"
}

# Recover a table left by an interrupted StePS run using only that run's
# prewritten provenance manifest. Unmanifested legacy tables are not adopted.
vlib::steps::recover_ewald_cache() {
    local stage_dir="${1}"
    local cache_root="${2}"
    vlib::steps::_ewald_cache_helper recover \
        --stage-dir "${stage_dir}" \
        --cache-root "${cache_root}"
}

# Run StePS with validated S^1 x R^2 Ewald staging, durable attempt logs, and
# a narrowly-scoped retry for StePS's missing-file false-positive.
# Usage: vlib::steps::run_binary_with_ewald_cache \
#   <binary> <param> <stage_dir> <cache_root> <stage_name> <mode>
vlib::steps::run_binary_with_ewald_cache() {
    local binary="${1}"
    local param="${2}"
    local stage_dir="${3}"
    local cache_root="${4}"
    local stage_name="${5}"
    local mode="${6}"
    local table="${stage_dir}/S1R2_Ewald_table_${mode}.hdf5"

    vlib::steps::_ewald_cache_helper prepare \
        --binary "${binary}" \
        --param "${param}" \
        --mode "${mode}" \
        --stage-dir "${stage_dir}" \
        --cache-root "${cache_root}" \
        --stage-name "${stage_name}"

    local log_root="${cache_root}/logs/${stage_name}"
    mkdir -p "${log_root}"
    local log_dir
    log_dir="$(mktemp -d "${log_root}/$(date -u '+%Y%m%dT%H%M%SZ')-$$-XXXXXX")"

    local attempt run_status promote_status
    local -a pipeline_status
    for attempt in 1 2 3 4 5; do
        local attempt_log="${log_dir}/attempt-${attempt}.log"
        echo "  StePS attempt ${attempt}/5 (log: ${attempt_log})"
        if vlib::steps::run_binary "${binary}" "${param}" 2>&1 | tee "${attempt_log}"; then
            pipeline_status=("${PIPESTATUS[@]}")
        else
            pipeline_status=("${PIPESTATUS[@]}")
        fi
        run_status="${pipeline_status[0]}"
        if (( pipeline_status[1] != 0 )); then
            echo "ERROR: Failed to write StePS attempt log ${attempt_log}." >&2
            return "${pipeline_status[1]}"
        fi

        promote_status=0
        vlib::steps::_ewald_cache_helper promote \
            --stage-dir "${stage_dir}" \
            --cache-root "${cache_root}" \
            --attempt-log "${attempt_log}" \
            || promote_status=$?

        if (( run_status == 0 )); then
            if (( promote_status != 0 )); then
                echo "ERROR: StePS succeeded but its Ewald table could not be preserved." >&2
                return "${promote_status}"
            fi
            return 0
        fi

        if (( promote_status != 0 && promote_status != 3 )); then
            echo "WARNING: The failed StePS run's Ewald table was not cacheable." >&2
        fi

        if (( attempt < 5 )) \
        && vlib::steps::_ewald_cache_helper check-missing-error \
            --log "${attempt_log}" --table "${table}"; then
            echo "  Retrying after StePS misreported the absent Ewald table as present."
            continue
        fi
        return "${run_status}"
    done
    return "${run_status}"
}

# Compute gravitational softening from a StePS snapshot via Python.
# Usage: softening="$(vlib::steps::compute_softening <snapshot.hdf5>)"
vlib::steps::compute_softening() {
    local snapshot="${1}"
    conda run --no-capture-output -n "${STEPSIC_ENV}" python - <<PYEOF
import h5py, numpy as np, sys
with h5py.File("${snapshot}", "r") as f:
    pos = f["Coordinates"][:]
n = len(pos)
lbox = float(np.max(pos) - np.min(pos))
mean_sep = lbox / n ** (1.0 / 3.0)
print(f"{mean_sep / 40.0:.6f}")
PYEOF
}

# ============================================================================
# Stepsic TOML generator
# ============================================================================
#
# Write a stepsic configuration TOML file.
# Usage: vlib::stepsic::write_toml <output_path> [KEY=VALUE ...]
#
# Cosmology name comes from COSMOLOGY_NAME (set by vlib::cosmology::load).
# Geometry parameters (R_3D, D_4D, RCRIT, NRBINS, NSHELL, etc.) come from
# the calling environment; override any by passing KEY=VALUE pairs.
#
# Example:
#   vlib::stepsic::write_toml "${TOML_DIR}/ic.toml" \
#       GEOMETRY=spherical LBOX=500 R_3D=500 IC_DIR="${CACHE}/ic"
vlib::stepsic::write_toml() {
    local path="${1}"
    shift

    # Collect explicit key=value overrides into an associative array.
    declare -A _t
    local pair
    for pair in "$@"; do
        _t["${pair%%=*}"]="${pair#*=}"
    done

    # Resolve a key: override wins, then env-var fallback, then hard default.
    local _tv
    _tv() {
        local k="${1}" d="${2:-}"
        if [[ -n "${_t[${k}]+x}" ]]; then
            echo "${_t[${k}]}"
        else
            local ev="${!k:-}"
            echo "${ev:-${d}}"
        fi
    }

    local GEOMETRY; GEOMETRY="$(_tv GEOMETRY spherical)"
    local LBOX;     LBOX="$(_tv LBOX 500)"
    local PERIODIC; PERIODIC="$(_tv PERIODIC false)"
    local R_3D_V;   R_3D_V="$(_tv R_3D "${R_3D:-500}")"
    local D_4D_V;   D_4D_V="$(_tv D_4D "${D_4D:-75}")"
    local RCRIT_V;  RCRIT_V="$(_tv RCRIT "${RCRIT:-0}")"
    local BIN_MODE; BIN_MODE="$(_tv BIN_MODE omega)"
    local NRBINS_V; NRBINS_V="$(_tv NRBINS "${NRBINS:-224}")"
    local IC_TYPE;  IC_TYPE="$(_tv TYPE preglass)"
    local NGRID_V;  NGRID_V="$(_tv NGRID "${NGRID:-32}")"
    local NPART_V;  NPART_V="$(_tv NPART "${NPART:-0}")"
    local NSHELL_V; NSHELL_V="$(_tv NSHELL "${NSHELL:-12288}")"
    local IC_DIR_V; IC_DIR_V="$(_tv IC_DIR "${IC_DIR:-output}")"
    local IC_PFX;   IC_PFX="$(_tv IC_PREFIX stepsic)"
    local SPECTRUM; SPECTRUM="$(_tv SPECTRUM camb)"
    local NONLINEAR;NONLINEAR="$(_tv NONLINEAR false)"
    local HALOFIT;  HALOFIT="$(_tv HALOFIT mead2020)"
    local SEED_V;   SEED_V="$(_tv SEED 42)"
    local LPTORDER; LPTORDER="$(_tv LPTORDER 0)"
    local NMESH_V;  NMESH_V="$(_tv NMESH 32)"
    local REDSHIFT; REDSHIFT="$(_tv REDSHIFT 0)"
    local INTERP;   INTERP="$(_tv INTERPOLATION cic)"
    local COMPENSATE; COMPENSATE="$(_tv COMPENSATE false)"
    local SPHEREMODE; SPHEREMODE="$(_tv SPHEREMODE false)"
    local PAIRED_V; PAIRED_V="$(_tv PAIRED false)"
    # Optional cosmological overrides (empty = use cosmology.toml defaults).
    # Set OMEGA_M=1.0 and OMEGA_L=0.0 for EdS pre-glass ICs so that particle
    # masses satisfy StePS's h-independent mass-consistency check (rho_part /
    # rho_cosm = 1).  When empty, the COSMOLOGY block in cosmology.toml supplies
    # the values via setdefault and these lines are omitted from the TOML.
    local OMEGA_M_V; OMEGA_M_V="$(_tv OMEGA_M "")"
    local OMEGA_L_V; OMEGA_L_V="$(_tv OMEGA_L "")"

    mkdir -p "$(dirname "${path}")"
    cat > "${path}" <<EOF2
GEOMETRY = "${GEOMETRY}"
LBOX = ${LBOX}
PERIODIC = ${PERIODIC}
R_3D = ${R_3D_V}
D_4D = ${D_4D_V}
RCRIT = ${RCRIT_V}
BIN_MODE = "${BIN_MODE}"
NRBINS = ${NRBINS_V}
COI = [0, 0, 0]
TYPE = "${IC_TYPE}"
NGRID = ${NGRID_V}
NPART = ${NPART_V}
NSHELL = ${NSHELL_V}
INPUT_GLASS = "none"
IC_DIR = "${IC_DIR_V}"
IC_PREFIX = "${IC_PFX}"
COSMOLOGY = "${COSMOLOGY_NAME:-Planck2018}"
${OMEGA_M_V:+OMEGA_M = ${OMEGA_M_V}}
${OMEGA_L_V:+OMEGA_L = ${OMEGA_L_V}}
SPECTRUM = "${SPECTRUM}"
NONLINEAR = ${NONLINEAR}
HALOFIT = "${HALOFIT}"
SEED = ${SEED_V}
COMOVING = true
HINDEPENDENT = true
LPTORDER = ${LPTORDER}
NMESH = ${NMESH_V}
REDSHIFT = ${REDSHIFT}
INTERPOLATION = "${INTERP}"
COMPENSATE = ${COMPENSATE}
SPHEREMODE = ${SPHEREMODE}
PAIRED = ${PAIRED_V}
PHASE_SHIFT = 0.0
NMESHSAMPLES = 1
ROTATE = 0.0
IC_FORMAT = "hdf5"
USE_DOUBLE = false
SAVE_WHITE_NOISE = false
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF2

    unset -f _tv
}

# Write a StePS parameter file for glass relaxation using EdS cosmology.
# Usage: vlib::steps::write_glass_param <name> <ic_file> <out_dir>
#            <is_periodic> <l_box> <r_sim> <softening>
#            <radial_force_accuracy> <radial_force_table_size>
# Reads from env: EDS_H0, GLASS_A_START, GLASS_A_MAX, GLASS_ACC_PARAM,
#   GLASS_STEP_MIN, GLASS_STEP_MAX, GLASS_TIME_LIMIT_MIN,
#   GLASS_FIRST_T_OUT (StePS FIRST_T_OUT, default 999.0),
#   GLASS_H_OUT       (StePS H_OUT,       default 999.0),
#   PARAM_DIR.
vlib::steps::write_glass_param() {
    local name="${1}"
    local ic_file="${2}"
    local out_dir="${3}"
    local is_periodic="${4}"
    local l_box="${5}"
    local r_sim="${6}"
    local soft="${7}"
    local rf_accuracy="${8}"
    local rf_table_size="${9}"

    mkdir -p "${PARAM_DIR}"
    cat > "${PARAM_DIR}/${name}.param" <<EOF2
Cosmological parameters:
------------------------
Omega_b         0.0
Omega_lambda    0.0
Omega_m         1.0
Omega_r         0.0
HubbleConstant  ${EDS_H0}
a_start         ${GLASS_A_START}
a_max           ${GLASS_A_MAX}

Simulation parameters:
-----------------------
COSMOLOGY       1
IS_PERIODIC     ${is_periodic}
COMOVING_INTEGRATION    1
L_BOX           ${l_box}
R_SIM           ${r_sim}
IC_FILE         ${ic_file}
IC_FORMAT       2
OUT_DIR         ${out_dir}
OUTPUT_FORMAT   2
OUTPUT_TIME_VARIABLE    1
ACC_PARAM       ${GLASS_ACC_PARAM}
STEP_MIN        ${GLASS_STEP_MIN}
STEP_MAX        ${GLASS_STEP_MAX}
PARTICLE_RADII  ${soft}
OUT_LST         /dev/null
FIRST_T_OUT     ${GLASS_FIRST_T_OUT:-999.0}
H_OUT           ${GLASS_H_OUT:-999.0}
SNAPSHOT_START_NUMBER   0
H_INDEPENDENT_UNITS     1
TIME_LIMIT_IN_MIN       ${GLASS_TIME_LIMIT_MIN}
RADIAL_FORCE_ACCURACY   ${rf_accuracy}
RADIAL_FORCE_TABLE_SIZE ${rf_table_size}
EOF2
}
