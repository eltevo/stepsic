#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  validation/cylinder/run.sh - Cylindrical (S^1 x R^2) end-to-end validation
#
#  End-to-end pipeline: glass -> 1LPT + 2LPT cosmological ICs -> N-body
#  simulations -> P(k) measurement and comparison plots.
#
#  Pipeline steps:
#    1. preglass      - cylindrical pre-glass IC  (stepsic, LPTORDER=0)
#    2. glass_param   - compute softening + write StePS glass parameter file
#    3. build_glass   - compile StePS binary  (PERIODIC_Z + GLASS_MAKING)
#    4. run_glass     - run glass relaxation
#    5. ic_2lpt       - 2LPT cosmological IC from glass
#    6. ic_1lpt       - 1LPT cosmological IC from glass (for LPT order comparison)
#    7. build_sim     - compile StePS LCDM binary  (PERIODIC_Z)
#    8. run_sim_2lpt  - run 2LPT LCDM simulation  (z_init -> z=0)
#    9. run_sim_1lpt  - run 1LPT LCDM simulation  (z_init -> z=0)
#   10. measure       - randoms + 1LPT/2LPT P(k) archives
#   11. plot          - validation figures
#   12. evaluate      - scientific result contract
#
#  When GLASS_SNAP is set, steps 1-4 are omitted and the pipeline starts at
#  ic_2lpt. The glass content hash enters the IC and measurement manifests.
#
#  Glass-making uses EdS cosmology (Omega_m=1) with reversed gravity (GLASS_MAKING).
#  H0_EdS = 100 km/s/Mpc (h_EdS = 1); see vlib::cosmology::eds_h0 in
#  validation/_common/lib.sh.
#
#  Gravitational softening is computed from the pre-glass IC geometry by
#  scripts/compute-softening.py (RCRIT-zone volume / N^{1/3} / 40).
#
#  IC particle order is shuffled after generation for MPI load balance
#  (cylindrical stepsic writes radial shells; without shuffle one rank gets
#  all the dense core particles).
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --step=N      Start from step N (1–12, default 1)
#    --plot-only   Refresh measurement and plotting inputs
#    --force       Re-run all steps regardless of cached outputs
#    --clean       Delete cache/ before running
#    --config=PATH Source an alternative config.env
#    --list-steps  Print the step list and exit
#    -h, --help    Print this help text and exit
#
#  Configuration (edit config.env or export before running):
#    STEPS_SRC, STEPSIC_SRC, STEPSIC_PY, STEPS_ENV, STEPSIC_ENV
#    STEPS_BACKEND, N_MPI, N_GPU, OMP_NUM_THREADS
#    GLASS_SNAP
#    R_3D, D_4D, RCRIT, LZ, NRBINS, NSHELL, BIN_MODE
#    SIM_Z_INIT, SIM_LPTORDER, SIM_NMESH
#    COSMOLOGY_NAME, COSMO_H0, COSMO_OMEGA_M, COSMO_OMEGA_L, COSMO_OMEGA_B
#    GLASS_TIME_LIMIT_MIN, GLASS_A_START, GLASS_A_MAX
#    GLASS_ACC_PARAM, GLASS_STEP_MIN, GLASS_STEP_MAX
#    SIM_IS_PERIODIC, SIM_ACC_PARAM, SIM_STEP_MIN, SIM_STEP_MAX
#    SIM_TIME_LIMIT_MIN, SIM_RADIAL_FORCE_ACCURACY, SIM_RADIAL_FORCE_TABLE_SIZE
#    STEPS_PRECISION  (double | single; default double)
#    PK_NMESH, PK_P0, PK_NRADIAL_BINS, PK_NFKP_RADIAL_BINS
#    PK_RANDOMS_NFACTOR, PK_RANDOMS_SEED
#
#  Examples:
#    # Full end-to-end run with 4 GPUs
#    N_GPU=4 bash run.sh
#
#    # Resume from step 5 (glass done, generate cosmological ICs)
#    N_GPU=4 bash run.sh --step=5
#
#    # Use a pre-generated glass (pipeline starts at ic_2lpt)
#    GLASS_SNAP=/path/to/glass.hdf5 bash run.sh
#
#    # Replot from cached simulation data
#    bash run.sh --plot-only
# ============================================================================

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"
vlib::source_config "${CONFIG_FILE}"

if [[ -n "${GLASS_SNAP}" ]]; then
    vlib::declare_steps ic_2lpt ic_1lpt build_sim run_sim_2lpt run_sim_1lpt measure plot evaluate
else
    vlib::declare_steps preglass glass_param build_glass run_glass ic_2lpt ic_1lpt build_sim run_sim_2lpt run_sim_1lpt measure plot evaluate
fi
if (( VLIB_LIST_STEPS )); then
    vlib::list_steps
    exit 0
fi
if [[ -n "${GLASS_SNAP}" && ! -f "${GLASS_SNAP}" ]]; then
    echo "ERROR: GLASS_SNAP not found: ${GLASS_SNAP}" >&2
    exit 1
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"
vlib::steps::validate_backend
if [[ -n "${GLASS_SNAP}" ]]; then
    vlib::manifest_input ic_2lpt "${GLASS_SNAP}"
    vlib::manifest_input ic_1lpt "${GLASS_SNAP}"
    vlib::manifest_input measure "${GLASS_SNAP}"
fi

# -- Directory layout --------------------------------------------------------
export VLIB_CACHE_DIR="${VLIB_CACHE_DIR:-${BASEDIR}/cache}"
PARAM_DIR="${BASEDIR}/configs"
BUILD_DIR="${BASEDIR}/builds"
OUTPUT="${BASEDIR}/output"
export PARAM_DIR BUILD_DIR

PREGLASS_DIR="${VLIB_CACHE_DIR}/preglass"
GLASS_DIR="${VLIB_CACHE_DIR}/glass"
IC_2LPT_DIR="${VLIB_CACHE_DIR}/ics_2lpt"
IC_1LPT_DIR="${VLIB_CACHE_DIR}/ics_1lpt"
SIM_2LPT_DIR="${VLIB_CACHE_DIR}/sim_2lpt"
SIM_1LPT_DIR="${VLIB_CACHE_DIR}/sim_1lpt"
PK_DIR="${VLIB_CACHE_DIR}/pk"
RANDOMS_DIR="${VLIB_CACHE_DIR}/randoms"

mkdir -p "${PARAM_DIR}" "${BUILD_DIR}" "${OUTPUT}" \
         "${PREGLASS_DIR}" "${GLASS_DIR}" \
         "${IC_2LPT_DIR}" "${IC_1LPT_DIR}" \
         "${SIM_2LPT_DIR}" "${SIM_1LPT_DIR}" \
         "${PK_DIR}" "${RANDOMS_DIR}"

# -- Derived values ----------------------------------------------------------
SIM_A_START="$(awk "BEGIN {printf \"%.15f\", 1.0 / (1.0 + ${SIM_Z_INIT})}")"
export EDS_H0
EDS_H0="$(vlib::cosmology::eds_h0 "${COSMO_H0}" "${COSMO_OMEGA_M}")"

# Precision build flag (empty for the default double)
STEPS_PRECISION_FLAGS="$(vlib::steps::precision_flags)"
STEPS_BACKEND_SUFFIX=""
STEPS_BACKEND_FLAGS=()
if [[ "${STEPS_BACKEND}" == "bh" ]]; then
    STEPS_BACKEND_SUFFIX="_bh"
    STEPS_BACKEND_FLAGS=("USE_BH=0.25" "RANDOMIZE_BH=123456")
fi
# Binary names carry the precision suffix so that switching STEPS_PRECISION
# or backend triggers a rebuild instead of silently reusing another binary.
GLASS_BIN="${BUILD_DIR}/StePS_glass_cylindrical${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"
SIM_BIN="${BUILD_DIR}/StePS_cylindrical${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"


# -- Conda envs --------------------------------------------------------------
vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"
if ! (( VLIB_PLOT_ONLY )); then
    vlib::ensure_env "${STEPS_ENV}"
fi

# -- Banner ------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  Cylindrical (S^1 x R^2) end-to-end validation pipeline"
echo "========================================================================"
echo ""
echo "  Geometry:       R_3D=${R_3D} Mpc, D_4D=${D_4D} Mpc, Lz=${LZ} Mpc"
echo "  Cosmology:      H0=${COSMO_H0}, Omega_m=${COSMO_OMEGA_M}, Omega_L=${COSMO_OMEGA_L}"
if [[ -n "${GLASS_SNAP}" ]]; then
    echo "  Glass:          pre-generated ${GLASS_SNAP}"
else
    echo "  Glass (EdS):    H0_EdS=${EDS_H0}, a=${GLASS_A_START} -> ${GLASS_A_MAX}"
fi
echo "  Simulation:     z_init=${SIM_Z_INIT} (a=${SIM_A_START}), LPT orders: 1 + ${SIM_LPTORDER}"
echo ""

# Derive softening for a supplied glass before running the consuming steps.
if [[ -n "${GLASS_SNAP}" ]]; then
    if [[ -z "${PARTICLE_RADII:-}" ]]; then
        PARTICLE_RADII="$(vlib::run_python "${STEPSIC_ENV}" \
            "${BASEDIR}/scripts/compute-softening.py" "${GLASS_SNAP}")"
    fi
    echo "  PARTICLE_RADII: ${PARTICLE_RADII}"
fi

# -- Helpers -----------------------------------------------------------------

# Write a cosmological IC TOML for cylindrical geometry (glass input).
_write_cosmo_toml() {
    local path="${1}" ic_dir="${2}" glass_snap="${3}" lptorder="${4}"
    local local_diam=$(( R_3D * 2 ))
    vlib::atomic_text "${path}" <<EOF
GEOMETRY = "cylindrical"
LBOX = [${local_diam}, ${local_diam}, ${LZ}]
PERIODIC = [0, 0, 1]
R_3D = ${R_3D}
D_4D = ${D_4D}
RCRIT = ${RCRIT}
BIN_MODE = "${BIN_MODE}"
NRBINS = ${NRBINS}
COI = [0, 0, 0]
TYPE = "glass"
NSHELL = ${NSHELL}
INPUT_GLASS = "${glass_snap}"
IC_DIR = "${ic_dir}"
IC_PREFIX = "stepsic"
COSMOLOGY = "${COSMOLOGY_NAME}"
SPECTRUM = "camb"
NONLINEAR = false
HALOFIT = "mead2020"
SEED = 42
COMOVING = true
HINDEPENDENT = true
LPTORDER = ${lptorder}
NMESH = ${SIM_NMESH}
REDSHIFT = ${SIM_Z_INIT}
INTERPOLATION = "cic"
COMPENSATE = false
SPHEREMODE = false
PAIRED = false
FIXED = false
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
EOF
}

# Shuffle IC particle order for MPI load balance; prints the final IC path.
_shuffle_ic() {
    local dir="${1}"
    local ic
    ic="$(vlib::find_ic "${dir}")"
    local script="${STEPS_SRC}/tools/Snapshot_Format_Converters/shuffle_hdf5_snap.py"
    if [[ -f "${script}" ]]; then
        local shuffled="${ic%.hdf5}_shuffled.hdf5"
        echo "  Shuffling IC for MPI load balance..." >&2
        vlib::run_in_env "${STEPSIC_ENV}" python "${script}" \
            -i "${ic}" -o "${shuffled}" -s 137 >&2
        echo "${shuffled}"
    else
        echo "  WARNING: shuffle_hdf5_snap.py not found; proceeding without shuffle." >&2
        echo "${ic}"
    fi
}

# Write the simulation output-redshift list.
_write_outredshifts() {
    vlib::atomic_text "${PARAM_DIR}/outredshifts.txt" <<'EOF'
10.0
5.0
3.0
2.0
1.0
0.5
0.0
EOF
}

# Write a StePS LCDM simulation parameter file.
# _write_lcdm_param <name> <ic_file> <out_dir>
_write_lcdm_param() {
    local name="${1}" ic_file="${2}" out_dir="${3}"
    # StePS reads R_SIM in physical Mpc: H_INDEPENDENT_UNITS=1 converts L_BOX and
    # PARTICLE_RADII from [Mpc/h] but not Rsim (see vlib::cosmology::eds_h0), so
    # convert R_3D [Mpc/h] here or the PERIODIC_Z mass check fails with 1/h^2.
    local r_sim_mpc
    r_sim_mpc="$(awk "BEGIN {printf \"%.10f\", ${R_3D} * 100.0 / ${COSMO_H0}}")"
    vlib::atomic_text "${PARAM_DIR}/lcdm_${name}.param" <<EOF
Cosmological parameters:
------------------------
Omega_b         ${COSMO_OMEGA_B}
Omega_lambda    ${COSMO_OMEGA_L}
Omega_m         ${COSMO_OMEGA_M}
Omega_r         0.0
HubbleConstant  ${COSMO_H0}
a_start         ${SIM_A_START}
a_max           1.0

Simulation parameters:
-----------------------
COSMOLOGY       1
IS_PERIODIC     ${SIM_IS_PERIODIC}
COMOVING_INTEGRATION    1
L_BOX           ${LZ}
R_SIM           ${r_sim_mpc}
IC_FILE         ${ic_file}
IC_FORMAT       2
OUT_DIR         ${out_dir}/
OUT_LST         ${PARAM_DIR}/outredshifts.txt
OUTPUT_TIME_VARIABLE    1
OUTPUT_FORMAT   2
REDSHIFT_CONE   0
MIN_REDSHIFT    0.0003012504
ACC_PARAM       ${SIM_ACC_PARAM}
RADIAL_FORCE_ACCURACY   ${SIM_RADIAL_FORCE_ACCURACY}
RADIAL_FORCE_TABLE_SIZE ${SIM_RADIAL_FORCE_TABLE_SIZE}
STEP_MIN        ${SIM_STEP_MIN}
STEP_MAX        ${SIM_STEP_MAX}
PARTICLE_RADII  ${PARTICLE_RADII}
FIRST_T_OUT     1.0
H_OUT           1.0
SNAPSHOT_START_NUMBER   0
H_INDEPENDENT_UNITS     1
TIME_LIMIT_IN_MIN       ${SIM_TIME_LIMIT_MIN}
EOF
}

# ============================================================================
# Steps
# ============================================================================

# Build a glass only when one was not supplied.
if [[ -z "${GLASS_SNAP}" ]]; then

# -- Step 1: preglass --------------------------------------------------------
if vlib::step_check "preglass" "${PREGLASS_DIR}"; then
    vlib::clear_dir "${PREGLASS_DIR}"
    local_diam=$(( R_3D * 2 ))
    vlib::stepsic::write_toml "${PARAM_DIR}/preglass.toml" \
        "GEOMETRY=cylindrical" \
        "LBOX=[${local_diam}, ${local_diam}, ${LZ}]" \
        "PERIODIC=[0, 0, 1]" \
        "R_3D=${R_3D}" \
        "D_4D=${D_4D}" \
        "RCRIT=${RCRIT}" \
        "BIN_MODE=${BIN_MODE}" \
        "NRBINS=${NRBINS}" \
        "TYPE=shell" \
        "NSHELL=${NSHELL}" \
        "IC_DIR=${PREGLASS_DIR}" \
        "LPTORDER=0" \
        "NMESH=${SIM_NMESH}" \
        "OMEGA_M=1.0" \
        "OMEGA_L=0.0"
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" "${PARAM_DIR}/preglass.toml"
    vlib::step_done "preglass"
fi

# -- Step 2: glass_param (softening computation + glass parameter file) ------
if vlib::step_check "glass_param" "${PARAM_DIR}/glass.param"; then
    IC_PREGLASS="$(vlib::find_ic "${PREGLASS_DIR}")"
    PARTICLE_RADII="$(vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/compute-softening.py" "${IC_PREGLASS}")"
    echo "  Pre-glass IC:   ${IC_PREGLASS}"
    echo "  PARTICLE_RADII: ${PARTICLE_RADII}"
    vlib::breadcrumb_set "IC_PREGLASS"    "${IC_PREGLASS}"
    vlib::breadcrumb_set "PARTICLE_RADII" "${PARTICLE_RADII}"
    # S^1 x R^2: IS_PERIODIC=4 (high-precision Ewald), L_BOX=Lz, R_SIM=R_3D.
    vlib::steps::write_glass_param \
        "glass" "${IC_PREGLASS}" "${GLASS_DIR}/" \
        4 "${LZ}" "${R_3D}" \
        "${PARTICLE_RADII}" \
        "1000" "1000"
    vlib::step_done "glass_param"
fi

IC_PREGLASS="${IC_PREGLASS:-$(vlib::breadcrumb_get IC_PREGLASS)}"
PARTICLE_RADII="${PARTICLE_RADII:-$(vlib::breadcrumb_get PARTICLE_RADII)}"

# -- Step 3: build_glass -----------------------------------------------------
if vlib::step_check "build_glass" "${GLASS_BIN}"; then
    vlib::steps::detect_toolchain
    vlib::steps::build "$(basename "${GLASS_BIN}")" \
        PERIODIC_Z GLASS_MAKING "${STEPS_BACKEND_FLAGS[@]}" ${STEPS_PRECISION_FLAGS}
    vlib::step_done "build_glass"
fi

# Write glass.param from the resolved environment so overrides
# such as GLASS_TIME_LIMIT_MIN apply independently of the glass_param cache.
vlib::steps::write_glass_param \
    "glass" "${IC_PREGLASS}" "${GLASS_DIR}/" \
    4 "${LZ}" "${R_3D}" \
    "${PARTICLE_RADII}" \
    "1000" "1000"

# -- Step 4: run_glass -------------------------------------------------------
if vlib::step_check "run_glass"; then
    vlib::clear_dir "${GLASS_DIR}"
    vlib::steps::run_binary_with_fresh_ewald \
        "${GLASS_BIN}" "${PARAM_DIR}/glass.param" \
        "${GLASS_DIR}" "S1R2_Ewald_table_higres.hdf5"
    GLASS_SNAP="$(vlib::find_last_snap "${GLASS_DIR}")"
    if [[ -z "${GLASS_SNAP}" ]]; then
        echo "ERROR: Glass relaxation produced no snapshots." >&2
        exit 1
    fi
    echo "  Glass snapshot: ${GLASS_SNAP}"
    vlib::breadcrumb_set "GLASS_SNAP" "${GLASS_SNAP}"
    vlib::step_done "run_glass"
fi

GLASS_SNAP="${GLASS_SNAP:-$(vlib::breadcrumb_get GLASS_SNAP)}"

fi

# -- Step 5: ic_2lpt ---------------------------------------------------------
if vlib::step_check "ic_2lpt" "${IC_2LPT_DIR}"; then
    if [[ -z "${GLASS_SNAP}" || ! -f "${GLASS_SNAP}" ]]; then
        echo "ERROR: No glass snapshot found; run from step 4 or earlier." >&2; exit 1
    fi
    vlib::clear_dir "${IC_2LPT_DIR}"
    _write_cosmo_toml "${PARAM_DIR}/cosmo_2lpt.toml" \
        "${IC_2LPT_DIR}" "${GLASS_SNAP}" "${SIM_LPTORDER}"
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" "${PARAM_DIR}/cosmo_2lpt.toml"
    IC_2LPT="$(_shuffle_ic "${IC_2LPT_DIR}")"
    vlib::breadcrumb_set "IC_2LPT" "${IC_2LPT}"
    vlib::step_done "ic_2lpt"
fi

IC_2LPT="${IC_2LPT:-$(vlib::breadcrumb_get IC_2LPT)}"

# -- Step 6: ic_1lpt ---------------------------------------------------------
if vlib::step_check "ic_1lpt" "${IC_1LPT_DIR}"; then
    if [[ -z "${GLASS_SNAP}" || ! -f "${GLASS_SNAP}" ]]; then
        echo "ERROR: No glass snapshot found; run from step 4 or earlier." >&2; exit 1
    fi
    vlib::clear_dir "${IC_1LPT_DIR}"
    _write_cosmo_toml "${PARAM_DIR}/cosmo_1lpt.toml" \
        "${IC_1LPT_DIR}" "${GLASS_SNAP}" 1
    vlib::run_in_env "${STEPSIC_ENV}" python "${STEPSIC_PY}" "${PARAM_DIR}/cosmo_1lpt.toml"
    IC_1LPT="$(_shuffle_ic "${IC_1LPT_DIR}")"
    vlib::breadcrumb_set "IC_1LPT" "${IC_1LPT}"
    vlib::step_done "ic_1lpt"
fi

IC_1LPT="${IC_1LPT:-$(vlib::breadcrumb_get IC_1LPT)}"

# -- Step 7: build_sim -------------------------------------------------------
if vlib::step_check "build_sim" "${SIM_BIN}"; then
    vlib::steps::detect_toolchain
    vlib::steps::build "$(basename "${SIM_BIN}")" \
        PERIODIC_Z "${STEPS_BACKEND_FLAGS[@]}" ${STEPS_PRECISION_FLAGS}
    vlib::step_done "build_sim"
fi

# -- Step 8: run_sim_2lpt ----------------------------------------------------
if vlib::step_check "run_sim_2lpt"; then
    if [[ -z "${IC_2LPT:-}" || ! -f "${IC_2LPT}" ]]; then
        echo "ERROR: No 2LPT IC; run from step 5 or earlier." >&2; exit 1
    fi
    if [[ -z "${PARTICLE_RADII:-}" ]]; then
        echo "ERROR: PARTICLE_RADII not set; run from step 2 or earlier." >&2; exit 1
    fi
    _write_outredshifts
    _write_lcdm_param "2lpt" "${IC_2LPT}" "${SIM_2LPT_DIR}"
    vlib::clear_dir "${SIM_2LPT_DIR}"
    vlib::steps::run_binary_with_fresh_ewald \
        "${SIM_BIN}" "${PARAM_DIR}/lcdm_2lpt.param" \
        "${SIM_2LPT_DIR}" "S1R2_Ewald_table_medres.hdf5"
    SNAP_2LPT="$(vlib::find_last_snap "${SIM_2LPT_DIR}")"
    if [[ -z "${SNAP_2LPT}" ]]; then
        echo "ERROR: 2LPT simulation produced no snapshots." >&2; exit 1
    fi
    vlib::breadcrumb_set "SNAP_2LPT" "${SNAP_2LPT}"
    vlib::step_done "run_sim_2lpt"
fi

SNAP_2LPT="${SNAP_2LPT:-$(vlib::breadcrumb_get SNAP_2LPT)}"

# -- Step 9: run_sim_1lpt ----------------------------------------------------
if vlib::step_check "run_sim_1lpt"; then
    if [[ -z "${IC_1LPT:-}" || ! -f "${IC_1LPT}" ]]; then
        echo "ERROR: No 1LPT IC; run from step 6 or earlier." >&2; exit 1
    fi
    if [[ -z "${PARTICLE_RADII:-}" ]]; then
        echo "ERROR: PARTICLE_RADII not set; run from step 2 or earlier." >&2; exit 1
    fi
    _write_outredshifts
    _write_lcdm_param "1lpt" "${IC_1LPT}" "${SIM_1LPT_DIR}"
    vlib::clear_dir "${SIM_1LPT_DIR}"
    vlib::steps::run_binary_with_fresh_ewald \
        "${SIM_BIN}" "${PARAM_DIR}/lcdm_1lpt.param" \
        "${SIM_1LPT_DIR}" "S1R2_Ewald_table_medres.hdf5"
    SNAP_1LPT="$(vlib::find_last_snap "${SIM_1LPT_DIR}")"
    if [[ -z "${SNAP_1LPT}" ]]; then
        echo "ERROR: 1LPT simulation produced no snapshots." >&2; exit 1
    fi
    vlib::breadcrumb_set "SNAP_1LPT" "${SNAP_1LPT}"
    vlib::step_done "run_sim_1lpt"
fi

SNAP_1LPT="${SNAP_1LPT:-$(vlib::breadcrumb_get SNAP_1LPT)}"

# -- Step 10: measure -------------------------------------------------------
# --plot-only measures spectra from cached simulation data before plotting.
_CYLINDER_PLOT_ONLY="${VLIB_PLOT_ONLY}"
if (( VLIB_PLOT_ONLY )); then
    VLIB_PLOT_ONLY=0
fi
if vlib::step_check "measure" "${PK_DIR}/pk_1lpt.txt" "${PK_DIR}/pk_2lpt.txt"; then
    for _req_var in GLASS_SNAP SNAP_2LPT SNAP_1LPT; do
        if [[ -z "${!_req_var:-}" || ! -f "${!_req_var}" ]]; then
            echo "ERROR: ${_req_var} is not set or file not found." >&2
            exit 1
        fi
    done

    _steps_pk="${STEPS_SRC}/tools/PowerSpectra/StePS_Pk.py"
    if [[ ! -f "${_steps_pk}" ]]; then
        echo "ERROR: StePS_Pk.py not found at ${_steps_pk}" >&2
        exit 1
    fi

    _stub_dir="$(mktemp -d)"
    trap 'rm -rf "${_stub_dir}"' EXIT
    printf '# stub\n' > "${_stub_dir}/pygadgetreader.py"
    printf '# stub\n' > "${_stub_dir}/glio.py"
    _pk_dir="$(dirname "${_steps_pk}")"
    _steps_ic_src="${_pk_dir}/../../StePS_IC/src"

    _randoms="${RANDOMS_DIR}/randoms.hdf5"
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/generate-randoms.py" \
        "${GLASS_SNAP}" "${_randoms}" \
        --nfactor "${PK_RANDOMS_NFACTOR}" \
        --seed "${PK_RANDOMS_SEED}"

    for order in 2 1; do
        if [[ "${order}" == 2 ]]; then
            snapshot="${SNAP_2LPT}"
        else
            snapshot="${SNAP_1LPT}"
        fi
        output_pk="${PK_DIR}/pk_${order}lpt.txt"
        PYTHONPATH="${_stub_dir}:${_pk_dir}:${_steps_ic_src}:${PYTHONPATH:-}" \
            vlib::run_python "${STEPSIC_ENV}" "${_steps_pk}" \
            "${snapshot}" "${_randoms}" "${GLASS_SNAP}" "${output_pk}" \
            --Geometry cylindrical \
            --n_radial_bins "${PK_NRADIAL_BINS}" \
            --n_FKP_radial_bins "${PK_NFKP_RADIAL_BINS}" \
            --Nmesh "${PK_NMESH}" \
            --P0 "${PK_P0}" \
            --ShotNoise \
            --verbose
    done
    vlib::step_done "measure"
fi
VLIB_PLOT_ONLY="${_CYLINDER_PLOT_ONLY}"

# -- Step 11: plot ----------------------------------------------------------
if vlib::step_check "plot" "${OUTPUT}/cylinder_pk.pdf" "${OUTPUT}/cylinder_lpt_ratio.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-pk-ratio.py" \
        --pk "${PK_DIR}/pk_2lpt.txt" \
        --snapshot "${SNAP_2LPT}" \
        -o "${OUTPUT}/cylinder_pk.pdf"
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-lpt-ratio.py" \
        --pk-1lpt "${PK_DIR}/pk_1lpt.txt" \
        --pk-2lpt "${PK_DIR}/pk_2lpt.txt" \
        -o "${OUTPUT}/cylinder_lpt_ratio.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --one-lpt "${PK_DIR}/pk_1lpt.txt" \
        --two-lpt "${PK_DIR}/pk_2lpt.txt" \
        --figure "${OUTPUT}/cylinder_pk.pdf" \
        --figure "${OUTPUT}/cylinder_lpt_ratio.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
