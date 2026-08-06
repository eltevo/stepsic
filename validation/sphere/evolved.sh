#!/usr/bin/env bash
# ==========================================================================
# Evolve one spherical StePS realization and compare it with one matched
# periodic realization using centered HEALPix shells and full-domain P(k).
#
# Usage:
#   bash evolved.sh [OPTIONS]
#
# Standard options:
#   --step=N / --plot-only / --force / --force-step=A,B / --clean
#   --config=PATH / --list-steps / -h / --help
#
# Required inputs:
#   REFERENCE_MANIFEST  matched-pair manifest from validation/reference-nbody/run.sh
#   GEOM_GLASS          complete relaxed spherical StePS glass
#
# Principal artifacts:
#   evolved-cache/sphere/pair-contract.json
#   evolved-cache/sphere/cl.npz
#   evolved-cache/sphere/pk/steps-native.txt
#   evolved-cache/sphere/pk/periodic-native.npz
#   evolved-cache/sphere/evolved-comparison.npz
#   output/evolved-sphere/evolved-cl.pdf
#   output/evolved-sphere/evolved-pk.pdf
#   output/evolved-sphere/result.json
#
# Steps:
#   1. geom_ic      matched spherical 2LPT IC and exact field-hash check
#   2. build        spherical StePS simulation binary
#   3. evolve       evolve the full StePS system to the requested epoch
#   4. contract     record final matched-pair epoch and resolution metadata
#   5. cl           matched-origin HEALPix spectra in configured shells
#   6. pk_randoms   random catalog following the complete glass selection
#   7. pk_steps     StePS_Pk.py over full snapshot, randoms, and glass
#   8. pk_periodic  periodic FFT estimator over the complete cube
#   9. compare      common-support and CAMB/Halofit diagnostics
#  10. plot_cl      evolved-cl.pdf
#  11. plot_pk      evolved-pk.pdf
#  12. evaluate     structural/provenance verdict (differences diagnostic)
# ==========================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
CONFIG_FILE="${VLIB_CONFIG_FILE:-${BASEDIR}/evolved-config.env}"
vlib::source_config "${CONFIG_FILE}"
vlib::declare_steps \
    geom_ic build evolve contract cl pk_randoms pk_steps pk_periodic \
    compare plot_cl plot_pk evaluate
if (( VLIB_LIST_STEPS )); then
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"
vlib::steps::validate_backend

if [[ -z "${REFERENCE_MANIFEST}" || ! -f "${REFERENCE_MANIFEST}" ]]; then
    echo "ERROR: REFERENCE_MANIFEST must name a periodic pair manifest." >&2
    exit 1
fi
if [[ -z "${GEOM_GLASS}" || ! -f "${GEOM_GLASS}" ]]; then
    echo "ERROR: GEOM_GLASS must name a complete relaxed spherical glass." >&2
    exit 1
fi

export VLIB_CACHE_DIR="${EVOLVED_CACHE_DIR:-${BASEDIR}/evolved-cache/sphere}"
PARAM_DIR="${BASEDIR}/evolved-configs/sphere"
BUILD_DIR="${BASEDIR}/evolved-builds"
OUTPUT="${BASEDIR}/output/evolved-sphere"
IC_DIR="${VLIB_CACHE_DIR}/ic"
SIM_DIR="${VLIB_CACHE_DIR}/simulation"
PK_DIR="${VLIB_CACHE_DIR}/pk"
PAIR_CONTRACT="${VLIB_CACHE_DIR}/pair-contract.json"
CL_DATA="${VLIB_CACHE_DIR}/cl.npz"
PK_RANDOMS="${PK_DIR}/randoms.hdf5"
STEPS_PK_NATIVE="${PK_DIR}/steps-native.txt"
PERIODIC_PK_NATIVE="${PK_DIR}/periodic-native.npz"
COMPARISON_DATA="${VLIB_CACHE_DIR}/evolved-comparison.npz"
CL_FIGURE="${OUTPUT}/evolved-cl.pdf"
PK_FIGURE="${OUTPUT}/evolved-pk.pdf"
EVOLVED_RESULT="${OUTPUT}/result.json"
export PARAM_DIR BUILD_DIR

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi
mkdir -p "${PARAM_DIR}" "${BUILD_DIR}" "${OUTPUT}" \
    "${IC_DIR}" "${SIM_DIR}" "${PK_DIR}"

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"
if ! (( VLIB_PLOT_ONLY )); then
    vlib::ensure_env "${STEPS_ENV}"
fi

LBOX_SIDE="$(awk "BEGIN {print 2*${R_3D}}")"
SIM_A_START="$(awk "BEGIN {printf \"%.15f\", 1.0/(1.0+${SIM_Z_INIT})}")"
SIM_A_FINAL="$(awk "BEGIN {printf \"%.15f\", 1.0/(1.0+${SIM_Z_FINAL})}")"
STEPS_PRECISION_FLAGS="$(vlib::steps::precision_flags)"
STEPS_BACKEND_SUFFIX=""
STEPS_BACKEND_FLAGS=()
if [[ "${STEPS_BACKEND}" == "bh" ]]; then
    STEPS_BACKEND_SUFFIX="_bh"
    STEPS_BACKEND_FLAGS=("USE_BH=0.25" "RANDOMIZE_BH=123456")
fi
SIM_BIN="${BUILD_DIR}/StePS_evolved_sphere${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"
STEPSIC_REVISION="$(git -C "${STEPSIC_SRC}" rev-parse HEAD)"
STEPS_REVISION="$(git -C "${STEPS_SRC}" rev-parse HEAD)"

echo ""
echo "========================================================================"
echo "  Evolved spherical matched-pair comparison"
echo "========================================================================"
echo "  R_3D=${R_3D} Mpc/h  RCRIT=${RCRIT} Mpc/h  Lbox=${LBOX_SIDE} Mpc/h"
echo "  reference manifest: ${REFERENCE_MANIFEST}"
echo "  HEALPix: NSIDE=${CL_NSIDE}, lmax=${CL_LMAX}, shells=${CL_SHELLS:-scaled defaults}"

CONTRACT_ARGS=(
    --manifest "${REFERENCE_MANIFEST}"
    --output "${PAIR_CONTRACT}"
    --radius-mpc-h "${R_3D}"
    --seed "${SIM_SEED}"
    --nmesh "${SIM_NMESH}"
    --ngrid "${PAIR_NGRID}"
    --lpt-order "${SIM_LPTORDER}"
    --initial-redshift "${SIM_Z_INIT}"
    --final-redshift "${SIM_Z_FINAL}"
    --H0-km-s-mpc "${COSMO_H0}"
    --omega-b "${COSMO_OMEGA_B}"
    --omega-m "${COSMO_OMEGA_M}"
    --omega-lambda "${COSMO_OMEGA_L}"
    --spectrum "${SIM_SPECTRUM}"
    --nonlinear "${SIM_NONLINEAR}"
    --halofit "${SIM_HALOFIT}"
    --interpolation "${SIM_INTERPOLATION}"
    --compensate "${SIM_COMPENSATE}"
    --sphere-mode "${SIM_SPHEREMODE}"
    --paired "${SIM_PAIRED}"
    --phase-shift-rad "${SIM_PHASE_SHIFT}"
    --nmesh-samples "${SIM_NMESHSAMPLES}"
    --use-double "${SIM_USE_DOUBLE}"
    --stepsic-revision "${STEPSIC_REVISION}"
    --simulation-revision "${STEPS_REVISION}"
)

# Reject mismatched reference configurations before generating ICs,
# compiling StePS, or starting either expensive evolution/measurement work.
vlib::run_python "${STEPSIC_ENV}" \
    "${BASEDIR}/scripts/check-pair-contract.py" configuration \
    "${CONTRACT_ARGS[@]}"

if vlib::step_check "geom_ic" "$(vlib::breadcrumb_get GEOM_IC)"; then
    vlib::clear_dir "${IC_DIR}"
    vlib::stepsic::write_toml "${PARAM_DIR}/geometry.toml" \
        "GEOMETRY=spherical" \
        "LBOX=[${LBOX_SIDE}, ${LBOX_SIDE}, ${LBOX_SIDE}]" \
        "PERIODIC=[0, 0, 0]" \
        "R_3D=${R_3D}" \
        "D_4D=${D_4D}" \
        "RCRIT=${RCRIT}" \
        "BIN_MODE=${BIN_MODE}" \
        "NRBINS=${NRBINS}" \
        "TYPE=glass" \
        "NSHELL=${NSHELL}" \
        "INPUT_GLASS=${GEOM_GLASS}" \
        "IC_DIR=${IC_DIR}" \
        "H0=${COSMO_H0}" \
        "OMEGA_B=${COSMO_OMEGA_B}" \
        "OMEGA_M=${COSMO_OMEGA_M}" \
        "OMEGA_L=${COSMO_OMEGA_L}" \
        "SPECTRUM=${SIM_SPECTRUM}" \
        "NONLINEAR=${SIM_NONLINEAR}" \
        "HALOFIT=${SIM_HALOFIT}" \
        "SEED=${SIM_SEED}" \
        "LPTORDER=${SIM_LPTORDER}" \
        "NMESH=${SIM_NMESH}" \
        "REDSHIFT=${SIM_Z_INIT}" \
        "INTERPOLATION=${SIM_INTERPOLATION}" \
        "COMPENSATE=${SIM_COMPENSATE}" \
        "SPHEREMODE=${SIM_SPHEREMODE}" \
        "PAIRED=${SIM_PAIRED}" \
        "PHASE_SHIFT=${SIM_PHASE_SHIFT}" \
        "NMESHSAMPLES=${SIM_NMESHSAMPLES}" \
        "USE_DOUBLE=${SIM_USE_DOUBLE}" \
        "SAVE_WHITE_NOISE=true"
    vlib::run_in_env "${STEPSIC_ENV}" \
        python "${STEPSIC_PY}" "${PARAM_DIR}/geometry.toml"
    GEOM_IC="$(vlib::find_ic "${IC_DIR}")"
    vlib::breadcrumb_set "GEOM_IC" "${GEOM_IC}"
    vlib::step_done "geom_ic"
fi
GEOM_IC="${GEOM_IC:-$(vlib::breadcrumb_get GEOM_IC)}"
GEOM_FIELD_DIR="$(dirname "${GEOM_IC}")"
GEOM_WHITE_NOISE="${GEOM_FIELD_DIR}/ic_white_noise.hdf5"
GEOM_DELTA_K="${GEOM_FIELD_DIR}/ic_delta_k.hdf5"
vlib::run_python "${STEPSIC_ENV}" \
    "${BASEDIR}/scripts/check-pair-contract.py" fields \
    "${CONTRACT_ARGS[@]}" \
    --steps-white-noise "${GEOM_WHITE_NOISE}" \
    --steps-delta-k "${GEOM_DELTA_K}"

if [[ -z "${GEOM_SOFTENING}" ]]; then
    GEOM_SOFTENING="$(
        vlib::run_python "${STEPSIC_ENV}" \
            "${BASEDIR}/scripts/compute-evolved-softening.py" \
            "${GEOM_GLASS}" --geometry spherical --radius "${RCRIT}" \
            --divisor "${SOFTENING_DIVISOR}"
    )"
fi
vlib::breadcrumb_set "GEOM_SOFTENING" "${GEOM_SOFTENING}"

if vlib::step_check "build" "${SIM_BIN}"; then
    vlib::steps::detect_toolchain
    vlib::steps::build "$(basename "${SIM_BIN}")" \
        "${STEPS_BACKEND_FLAGS[@]}" ${STEPS_PRECISION_FLAGS}
    vlib::step_done "build"
fi

_write_outredshifts() {
    vlib::atomic_text "${PARAM_DIR}/outredshifts.txt" <<EOF
${SIM_Z_FINAL}
EOF
}

_write_sim_param() {
    local r_sim_mpc
    r_sim_mpc="$(awk "BEGIN {printf \"%.10f\", ${R_3D}*100.0/${COSMO_H0}}")"
    vlib::atomic_text "${PARAM_DIR}/simulation.param" <<EOF
Cosmological parameters:
------------------------
Omega_b         ${COSMO_OMEGA_B}
Omega_lambda    ${COSMO_OMEGA_L}
Omega_m         ${COSMO_OMEGA_M}
Omega_r         0.0
HubbleConstant  ${COSMO_H0}
a_start         ${SIM_A_START}
a_max           ${SIM_A_FINAL}

Simulation parameters:
-----------------------
COSMOLOGY       1
IS_PERIODIC     0
COMOVING_INTEGRATION    1
L_BOX           ${LBOX_SIDE}
R_SIM           ${r_sim_mpc}
IC_FILE         ${GEOM_IC}
IC_FORMAT       2
OUT_DIR         ${SIM_DIR}/
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
PARTICLE_RADII  ${GEOM_SOFTENING}
FIRST_T_OUT     ${SIM_A_FINAL}
H_OUT           1.0
SNAPSHOT_START_NUMBER   0
H_INDEPENDENT_UNITS     1
TIME_LIMIT_IN_MIN       ${SIM_TIME_LIMIT_MIN}
EOF
}

if vlib::step_check "evolve" "$(vlib::breadcrumb_get GEOM_SNAPSHOT)"; then
    _write_outredshifts
    _write_sim_param
    vlib::clear_dir "${SIM_DIR}"
    vlib::steps::run_binary "${SIM_BIN}" "${PARAM_DIR}/simulation.param"
    GEOM_SNAPSHOT="$(vlib::find_last_snap "${SIM_DIR}")"
    [[ -n "${GEOM_SNAPSHOT}" ]] || {
        echo "ERROR: StePS evolution produced no snapshot." >&2
        exit 1
    }
    vlib::breadcrumb_set "GEOM_SNAPSHOT" "${GEOM_SNAPSHOT}"
    vlib::step_done "evolve"
fi
GEOM_SNAPSHOT="${GEOM_SNAPSHOT:-$(vlib::breadcrumb_get GEOM_SNAPSHOT)}"

if vlib::step_check "contract" "${PAIR_CONTRACT}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/check-pair-contract.py" evolution \
        "${CONTRACT_ARGS[@]}" \
        --steps-white-noise "${GEOM_WHITE_NOISE}" \
        --steps-delta-k "${GEOM_DELTA_K}" \
        --steps-snapshot "${GEOM_SNAPSHOT}" \
        --steps-softening-mpc-h "${GEOM_SOFTENING}" \
        --steps-force-mesh "${SIM_RADIAL_FORCE_TABLE_SIZE}"
    vlib::step_done "contract"
fi

if vlib::step_check "cl" "${CL_DATA}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/measure-evolved-cl.py" \
        --contract "${PAIR_CONTRACT}" \
        --steps-snapshot "${GEOM_SNAPSHOT}" \
        --shells "${CL_SHELLS}" \
        --radius-mpc-h "${R_3D}" \
        --rcrit-mpc-h "${RCRIT}" \
        --nside "${CL_NSIDE}" \
        --lmax "${CL_LMAX}" \
        --output "${CL_DATA}"
    vlib::step_done "cl"
fi

if vlib::step_check "pk_randoms" "${PK_RANDOMS}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/prepare-evolved-randoms.py" \
        --glass "${GEOM_GLASS}" \
        --factor "${PK_RANDOM_FACTOR}" \
        --seed "${PK_RANDOM_SEED}" \
        --output "${PK_RANDOMS}"
    vlib::step_done "pk_randoms"
fi

if vlib::step_check "pk_steps" "${STEPS_PK_NATIVE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/run-evolved-fkp.py" \
        --snapshot "${GEOM_SNAPSHOT}" \
        --glass "${GEOM_GLASS}" \
        --randoms "${PK_RANDOMS}" \
        --steps-pk "${STEPS_SRC}/tools/PowerSpectra/StePS_Pk.py" \
        --p0 "${PK_P0}" \
        --n-fkp-radial-bins "${PK_NFKP_RADIAL_BINS}" \
        --nmesh "${PK_NMESH}" \
        --output "${STEPS_PK_NATIVE}" --verbose
    vlib::step_done "pk_steps"
fi

PERIODIC_SNAPSHOT="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["snapshot"])' \
        "${REFERENCE_MANIFEST}"
)"
if vlib::step_check "pk_periodic" "${PERIODIC_PK_NATIVE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/measure-evolved-periodic-pk.py" \
        --snapshot "${PERIODIC_SNAPSHOT}" \
        --nmesh "${PK_NMESH}" \
        --output "${PERIODIC_PK_NATIVE}"
    vlib::step_done "pk_periodic"
fi

if vlib::step_check "compare" "${COMPARISON_DATA}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/compare-evolved.py" \
        --contract "${PAIR_CONTRACT}" \
        --cl "${CL_DATA}" \
        --steps-pk "${STEPS_PK_NATIVE}" \
        --periodic-pk "${PERIODIC_PK_NATIVE}" \
        --random-factor "${PK_RANDOM_FACTOR}" \
        --random-seed "${PK_RANDOM_SEED}" \
        --p0 "${PK_P0}" \
        --n-fkp-radial-bins "${PK_NFKP_RADIAL_BINS}" \
        --nmesh "${PK_NMESH}" \
        --output "${COMPARISON_DATA}"
    vlib::step_done "compare"
fi

if vlib::step_check "plot_cl" "${CL_FIGURE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/plot-evolved-cl.py" \
        --input "${COMPARISON_DATA}" --output "${CL_FIGURE}"
    vlib::step_done "plot_cl"
fi

if vlib::step_check "plot_pk" "${PK_FIGURE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/plot-evolved-pk.py" \
        --input "${COMPARISON_DATA}" --output "${PK_FIGURE}"
    vlib::step_done "plot_pk"
fi

if vlib::step_check "evaluate" "${EVOLVED_RESULT}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/evaluate-evolved.py" \
        --archive "${COMPARISON_DATA}" \
        --contract "${PAIR_CONTRACT}" \
        --cl-figure "${CL_FIGURE}" \
        --pk-figure "${PK_FIGURE}" \
        --cl-native "${CL_DATA}" \
        --steps-pk-native "${STEPS_PK_NATIVE}" \
        --periodic-pk-native "${PERIODIC_PK_NATIVE}" \
        --output "${EVOLVED_RESULT}"
    vlib::step_done "evaluate"
fi

vlib::report_done
