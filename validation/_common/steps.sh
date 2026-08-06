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
# Run StePS after removing its run-local Ewald table. Retry once only for the
# known unchecked-stat false positive, where StePS tries to load that absent
# table and HDF5 reports ENOENT.
# Usage: vlib::steps::run_binary_with_fresh_ewald \
#   <binary> <param> <output_dir> <table_name>
vlib::steps::run_binary_with_fresh_ewald() {
    if (( $# != 4 )); then
        echo "ERROR: run_binary_with_fresh_ewald expects 4 arguments." >&2
        return 2
    fi

    local binary="${1}"
    local param="${2}"
    local output_dir="${3}"
    local table_name="${4}"
    case "${table_name}" in
        Ewald_table_lowres.hdf5|Ewald_table_medres.hdf5|Ewald_table_higres.hdf5|\
        S1R2_Ewald_table_lowres.hdf5|S1R2_Ewald_table_medres.hdf5|\
        S1R2_Ewald_table_higres.hdf5) ;;
        *)
            echo "ERROR: Unsupported StePS Ewald table name: ${table_name}" >&2
            return 2
            ;;
    esac
    if [[ ! -d "${output_dir}" ]]; then
        echo "ERROR: StePS output directory does not exist: ${output_dir}" >&2
        return 2
    fi

    local table="${output_dir%/}/${table_name}"
    local attempt_log
    attempt_log="$(mktemp)" || return

    local remove_status=0
    rm -f -- "${table}" || remove_status=$?
    if (( remove_status != 0 )); then
        rm -f -- "${attempt_log}"
        return "${remove_status}"
    fi
    local run_status
    local -a pipeline_status
    if vlib::steps::run_binary "${binary}" "${param}" 2>&1 | tee "${attempt_log}"; then
        pipeline_status=("${PIPESTATUS[@]}")
    else
        pipeline_status=("${PIPESTATUS[@]}")
    fi
    run_status="${pipeline_status[0]}"
    if (( pipeline_status[1] != 0 )); then
        rm -f -- "${attempt_log}"
        echo "ERROR: Failed to capture StePS output while checking the Ewald failure." >&2
        return "${pipeline_status[1]}"
    fi
    if (( run_status == 0 )); then
        rm -f -- "${attempt_log}"
        return 0
    fi

    local retry=0
    if [[ ! -e "${table}" && ! -L "${table}" ]] \
    && grep -Fq -- "Ewald lookup table file (${table}) found." "${attempt_log}" \
    && grep -Fq -- "name = '${table}', errno = 2, error message = 'No such file or directory'" "${attempt_log}" \
    && grep -Fq -- "HDF5: cannot open file ${table}" "${attempt_log}" \
    && grep -Fq -- "Error: Failed to load the Ewald lookup table from file ${table}" "${attempt_log}" \
    && grep -Fq -- "Aborting." "${attempt_log}"; then
        retry=1
    fi
    rm -f -- "${attempt_log}"

    if (( retry )); then
        local retry_status=0
        echo "  Retrying after StePS misreported the absent Ewald table as present."
        rm -f -- "${table}" || return
        vlib::steps::run_binary "${binary}" "${param}" || retry_status=$?
        return "${retry_status}"
    fi
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
    vlib::atomic_text "${PARAM_DIR}/${name}.param" <<EOF2
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
