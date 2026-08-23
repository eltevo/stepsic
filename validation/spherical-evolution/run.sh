#!/usr/bin/env bash
# ==========================================================================
# Evolve one spherical StePS model and compare it with a periodic model using centred HEALPix shells and full-domain P(k).
#
# Usage:
#   bash run.sh [OPTIONS]
#
# Standard options:
#   --size=SIZE / --step=N / --plot-only / --force / --force-step=A,B / --clean
#   --config=PATH / --list-steps / -h / --help
#
# Required inputs:
#   REFERENCE_MANIFEST  periodic run description from validation/periodic-lpt/run.sh
#   GLASS_SNAP          complete relaxed spherical StePS glass
#                       (content hash included in consuming step manifests)
#
# Outputs:
#   runs/<profile>/cache/pair-contract.json
#   runs/<profile>/cache/cl.npz
#   runs/<profile>/cache/pk/steps-native.txt
#   runs/<profile>/cache/pk/periodic-native.npz
#   runs/<profile>/cache/evolved-comparison.npz
#   runs/<profile>/output/supplementary/spherical-evolution-cl.pdf
#   runs/<profile>/output/supplementary/spherical-evolution-pk.pdf
#
# Steps:
#   1. geom_ic      make a spherical 2LPT IC and confirm that its field matches
#   2. build        spherical StePS simulation binary
#   3. evolve       evolve the full StePS system to the requested epoch
#   4. contract     confirm that final epochs and resolutions match
#   5. cl           measure HEALPix spectra in matching radial shells
#   6. pk_randoms   make random positions with the glass's radial distribution
#   7. pk_steps     measure the full StePS snapshot with StePS_Pk.py
#   8. pk_periodic  measure the complete periodic cube with an FFT
#   9. compare      compare both spectra with CAMB/Halofit on shared k values
#  10. plot_cl      evolved-cl.pdf
#  11. plot_pk      evolved-pk.pdf
# ==========================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${BASEDIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign spherical-evolution "${BASEDIR}" "${BASEDIR}/config.env"
vlib::declare_steps \
    geom_ic build evolve contract cl pk_randoms pk_steps pk_periodic \
    compare plot_cl plot_pk
vlib::manifest_implementation contract \
    "${BASEDIR}/scripts/check-pair-contract.py" \
    "${BASEDIR}/scripts/evolved.py"
vlib::manifest_implementation evolve \
    "${BASEDIR}/scripts/compute-evolved-softening.py"
vlib::manifest_implementation cl \
    "${BASEDIR}/scripts/measure-evolved-cl.py" \
    "${BASEDIR}/scripts/evolved.py"
vlib::manifest_implementation pk_randoms \
    "${BASEDIR}/scripts/prepare-evolved-randoms.py" \
    "${BASEDIR}/scripts/evolved.py"
vlib::manifest_implementation pk_steps \
    "${BASEDIR}/scripts/run-evolved-fkp.py"
vlib::manifest_implementation pk_periodic \
    "${BASEDIR}/scripts/measure-evolved-periodic-pk.py" \
    "${BASEDIR}/scripts/evolved.py"
vlib::manifest_implementation compare \
    "${BASEDIR}/scripts/compare-evolved.py" \
    "${BASEDIR}/scripts/evolved.py"
vlib::manifest_implementation plot_cl \
    "${BASEDIR}/scripts/plot-evolved-cl.py"
vlib::manifest_implementation plot_pk \
    "${BASEDIR}/scripts/plot-evolved-pk.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
if [[ -z "${REFERENCE_MANIFEST}" || ! -f "${REFERENCE_MANIFEST}" ]]; then
    echo "ERROR: REFERENCE_MANIFEST must name a periodic pair manifest." >&2
    exit 1
fi
case "${GLASS_INPUT_MODE}" in
    generate)
        if [[ "${VLIB_SIZE}" == custom || "${VLIB_SIZE}" == large ]]; then
            echo "ERROR: generated spherical glass requires a small or medium profile." >&2
            exit 2
        fi
        GENERATED_GLASS_ROOT="${VLIB_RUN_ROOT}/cache/generated-glass"
        GLASS_INPUT_MODE=generate GLASS_PHASE=generation-only \
            VLIB_RUN_ROOT_OVERRIDE="${GENERATED_GLASS_ROOT}" \
            VLIB_RUN_ROOT_PARENT="${VLIB_RUN_ROOT}" \
            bash "${BASEDIR}/../glass-quality/run.sh" --size="${VLIB_SIZE}" \
            --evaluation=skip --run=spherical
        GLASS_SNAP="$(vlib::find_last_snap "${GENERATED_GLASS_ROOT}/cache/spherical/glass")"
        ;;
    pre-generated) ;;
    *) echo "ERROR: GLASS_INPUT_MODE must be generate or pre-generated." >&2; exit 2 ;;
esac
if [[ -z "${GLASS_SNAP}" || ! -f "${GLASS_SNAP}" ]]; then
    echo "ERROR: pre-generated mode requires an existing GLASS_SNAP." >&2
    exit 2
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"
vlib::steps::validate_backend
vlib::manifest_input geom_ic "${GLASS_SNAP}"
vlib::manifest_input pk_randoms "${GLASS_SNAP}"
vlib::manifest_input pk_steps "${GLASS_SNAP}"

export VLIB_CACHE_DIR="${EVOLVED_CACHE_DIR:-${VLIB_RUN_ROOT}/cache}"
PARAM_DIR="${VLIB_RUN_ROOT}/config/generated"
BUILD_DIR="${VLIB_RUN_ROOT}/build"
OUTPUT="${VLIB_RUN_ROOT}/output"
FIGURES="${OUTPUT}/supplementary"
IC_DIR="${VLIB_CACHE_DIR}/ic"
SIM_DIR="${VLIB_CACHE_DIR}/simulation"
PK_DIR="${VLIB_CACHE_DIR}/pk"
PAIR_CONTRACT="${VLIB_CACHE_DIR}/pair-contract.json"
CL_DATA="${VLIB_CACHE_DIR}/cl.npz"
PK_RANDOMS="${PK_DIR}/randoms.hdf5"
STEPS_PK_NATIVE="${PK_DIR}/steps-native.txt"
PERIODIC_PK_NATIVE="${PK_DIR}/periodic-native.npz"
COMPARISON_DATA="${VLIB_CACHE_DIR}/evolved-comparison.npz"
CL_FIGURE="${FIGURES}/spherical-evolution-cl.pdf"
PK_FIGURE="${FIGURES}/spherical-evolution-pk.pdf"
export PARAM_DIR BUILD_DIR

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi
mkdir -p "${PARAM_DIR}" "${BUILD_DIR}" "${FIGURES}" \
    "${IC_DIR}" "${SIM_DIR}" "${PK_DIR}"

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"
vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/../_common/snapshots.py" \
    "${GLASS_SNAP}" --geometry spherical --radius-mpc-h "${R_3D}"
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
echo "  Evolved spherical and periodic comparison"
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
        "INPUT_GLASS=${GLASS_SNAP}" \
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
            "${GLASS_SNAP}" --geometry spherical --radius "${RCRIT}" \
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
        --glass "${GLASS_SNAP}" \
        --factor "${PK_RANDOM_FACTOR}" \
        --seed "${PK_RANDOM_SEED}" \
        --output "${PK_RANDOMS}"
    vlib::step_done "pk_randoms"
fi

if vlib::step_check "pk_steps" "${STEPS_PK_NATIVE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/run-evolved-fkp.py" \
        --snapshot "${GEOM_SNAPSHOT}" \
        --glass "${GLASS_SNAP}" \
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

vlib::report_done
