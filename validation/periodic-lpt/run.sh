#!/usr/bin/env bash
# ============================================================================
#  Evolve matched periodic 1LPT and 2LPT initial conditions with Gadget-4.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Standard options:
#    --size=SIZE / --step=N / --plot-only / --force / --force-step=A,B / --clean
#    --config=PATH / --list-steps / -h / --help
#
#  Steps:
#    1. ic_load   - unperturbed grid paired to the cosmological ICs
#    2. ic_2lpt   - periodic 2LPT stepsic IC
#    3. ic_1lpt   - periodic 1LPT stepsic IC (matched control)
#    4. build     - configuration-keyed Gadget-4 build
#    5. run_2lpt  - evolve the reference to z=0
#    6. run_1lpt  - evolve the matched control to z=0
#    7. publish   - record the periodic run for the spherical comparison
#    8. compare   - common periodic estimator for 1LPT and 2LPT
#    9. plot      - control-ratio figure
#
#  The z=0 2LPT snapshot and unperturbed load are consumed by
#  validation/spherical-evolution/run.sh. Units are Mpc/h, 1e11 Msun/h, km/s.
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign periodic-lpt "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps \
    ic_load ic_2lpt ic_1lpt build run_2lpt run_1lpt publish compare plot evaluate
vlib::manifest_implementation publish \
    "${BASEDIR}/scripts/write-pair-manifest.py"
vlib::manifest_implementation compare \
    "${BASEDIR}/scripts/measure-control.py"
vlib::manifest_implementation plot "${BASEDIR}/scripts/plot-control.py"
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

if [[ ! -d "${GADGET4_SRC}/.git" ]]; then
    echo "ERROR: Gadget-4 source not found at ${GADGET4_SRC}" >&2
    exit 1
fi
if [[ ! -f "${STEPSIC_PY}" ]]; then
    echo "ERROR: stepsic.py not found at ${STEPSIC_PY}" >&2
    exit 1
fi

REF_A_START="$(awk "BEGIN {printf \"%.15f\", 1.0 / (1.0 + ${REF_Z_INIT})}")"
REF_A_FINAL="$(awk "BEGIN {printf \"%.15f\", 1.0 / (1.0 + ${REF_Z_FINAL})}")"
if [[ -z "${REF_SOFTENING_MPC_H}" ]]; then
    REF_SOFTENING_MPC_H="$(
        awk "BEGIN {printf \"%.10g\", ${REF_LBOX_MPC_H} / ${REF_NGRID} / 40.0}"
    )"
fi

GADGET_REVISION="$(git -C "${GADGET4_SRC}" rev-parse HEAD)"
STEPSIC_REVISION="$(git -C "${STEPSIC_SRC}" rev-parse HEAD)"
REFERENCE_RUN_KEY="$(
    {
        printf "%s\n" \
            "${GADGET_REVISION}" "${STEPSIC_REVISION}" \
            "${REF_LBOX_MPC_H}" "${REF_NGRID}" "${REF_NMESH}" \
            "${REF_PMGRID}" "${REF_SEED}" "${REF_Z_INIT}" \
            "${REF_Z_FINAL}" \
            "${REF_RUN_1LPT}" "${REF_SOFTENING_MPC_H}" \
            "${REF_SPECTRUM}" "${REF_NONLINEAR}" "${REF_HALOFIT}" \
            "${REF_INTERPOLATION}" "${REF_COMPENSATE}" \
            "${REF_SPHEREMODE}" "${REF_PAIRED}" \
            "${REF_NMESHSAMPLES}" "${REF_USE_DOUBLE}" \
            "${COSMOLOGY_NAME}" "${COSMO_OMEGA_B}" "${COSMO_OMEGA_M}" \
            "${COSMO_OMEGA_L}" "${COSMO_H0}" \
            "${GADGET_ERR_TOL_INT}" "${GADGET_ERR_TOL_FORCE}" \
            "${GADGET_MAX_TIMESTEP}" "${PK_NMESH}" "${PK_KMAX_FRAC_NY}"
    } | sha256sum | cut -c1-16
)"
REFERENCE_CACHE_ROOT="${REFERENCE_CACHE_ROOT:-${VLIB_RUN_ROOT}/cache}"
export VLIB_CACHE_DIR="${REFERENCE_CACHE_ROOT}/${REFERENCE_RUN_KEY}"
CONFIG_DIR="${VLIB_RUN_ROOT}/config/generated/${REFERENCE_RUN_KEY}"
BUILD_DIR="${VLIB_RUN_ROOT}/build"
OUTPUT="${VLIB_RUN_ROOT}/output"
FIGURES="${OUTPUT}/supplementary"
IC_LOAD_DIR="${VLIB_CACHE_DIR}/ic_load"
IC_2LPT_DIR="${VLIB_CACHE_DIR}/ic_2lpt"
IC_1LPT_DIR="${VLIB_CACHE_DIR}/ic_1lpt"
RUN_2LPT_DIR="${VLIB_CACHE_DIR}/run_2lpt"
RUN_1LPT_DIR="${VLIB_CACHE_DIR}/run_1lpt"
GADGET_RUNTIME_DIR="${VLIB_CACHE_DIR}/gadget_runtime"
CONTROL_DATA="${VLIB_CACHE_DIR}/periodic-control.npz"
PAIR_MANIFEST="${VLIB_CACHE_DIR}/pair-manifest.json"
CONTROL_FIGURE="${FIGURES}/periodic-control.pdf"

mkdir -p "${CONFIG_DIR}" "${BUILD_DIR}" "${FIGURES}" \
    "${IC_LOAD_DIR}" "${IC_2LPT_DIR}" "${IC_1LPT_DIR}" \
    "${RUN_2LPT_DIR}" "${RUN_1LPT_DIR}" "${GADGET_RUNTIME_DIR}"

GADGET_CONFIG="${CONFIG_DIR}/Gadget4-reference.sh"
GADGET_PARAM_2LPT="${CONFIG_DIR}/reference-2lpt.param"
GADGET_PARAM_1LPT="${CONFIG_DIR}/reference-1lpt.param"
GADGET_OUTPUTS="${CONFIG_DIR}/outputs.txt"

_atomic_text() {
    local target="${1}"
    local temporary
    temporary="$(mktemp "$(dirname "${target}")/.${target##*/}.tmp-XXXXXX")"
    cat > "${temporary}"
    sync -f "${temporary}"
    mv -f "${temporary}" "${target}"
}

_write_gadget_config() {
    _atomic_text "${GADGET_CONFIG}" <<EOF
PERIODIC
SELFGRAVITY
RANDOMIZE_DOMAINCENTER
PMGRID=${REF_PMGRID}
TREEPM_NOTIMESPLIT
ASMTH=2.0
NSOFTCLASSES=1
NTYPES=6
POSITIONS_IN_64BIT
DOUBLEPRECISION=1
OUTPUT_IN_DOUBLEPRECISION
IDS_64BIT
GADGET2_HEADER
EOF
}

_write_output_list() {
    _atomic_text "${GADGET_OUTPUTS}" <<EOF
${REF_A_FINAL}
EOF
}

_write_gadget_param() {
    local target="${1}" ic_file="${2}" output_dir="${3}"
    ic_file="${ic_file%.hdf5}"
    _atomic_text "${target}" <<EOF
InitCondFile                    ${ic_file}
OutputDir                       ${output_dir}/
SnapshotFileBase                snapshot
OutputListFilename              ${GADGET_OUTPUTS}

ICFormat                        3
SnapFormat                      3

TimeLimitCPU                    ${GADGET_TIME_LIMIT_S}
CpuTimeBetRestartFile           ${GADGET_RESTART_INTERVAL_S}
MaxMemSize                      ${GADGET_MAX_MEM_MB}

TimeBegin                       ${REF_A_START}
TimeMax                         ${REF_A_FINAL}
ComovingIntegrationOn           1

Omega0                          ${COSMO_OMEGA_M}
OmegaLambda                     ${COSMO_OMEGA_L}
OmegaBaryon                     ${COSMO_OMEGA_B}
HubbleParam                     $(awk "BEGIN {printf \"%.12g\", ${COSMO_H0} / 100.0}")
Hubble                          100.0
BoxSize                         ${REF_LBOX_MPC_H}

OutputListOn                    1
TimeBetSnapshot                 0.0
TimeOfFirstSnapshot             0.0
TimeBetStatistics               0.01
NumFilesPerSnapshot             1
MaxFilesWithConcurrentIO        1

ErrTolIntAccuracy               ${GADGET_ERR_TOL_INT}
CourantFac                      0.3
MaxSizeTimestep                 ${GADGET_MAX_TIMESTEP}
MinSizeTimestep                 0.0

TypeOfOpeningCriterion          1
ErrTolTheta                     0.75
ErrTolThetaMax                  1.0
ErrTolForceAcc                  ${GADGET_ERR_TOL_FORCE}
TopNodeFactor                   3.0
ActivePartFracForNewDomainDecomp 0.01
ActivePartFracForPMinsteadOfEwald 0.05

DesNumNgb                       64
MaxNumNgbDeviation              1

UnitLength_in_cm                3.085678e24
UnitMass_in_g                   1.989e44
UnitVelocity_in_cm_per_s        1.0e5
GravityConstantInternal         0

SofteningComovingClass0         ${REF_SOFTENING_MPC_H}
SofteningMaxPhysClass0          ${REF_SOFTENING_MPC_H}
SofteningClassOfPartType0       0
SofteningClassOfPartType1       0
SofteningClassOfPartType2       0
SofteningClassOfPartType3       0
SofteningClassOfPartType4       0
SofteningClassOfPartType5       0

ArtBulkViscConst                1.0
MinEgySpec                      0
InitGasTemp                     0
EOF
}

_write_stepsic_ic() {
    local output_dir="${1}" lpt_order="${2}" config_name="${3}"
    vlib::clear_dir "${output_dir}"
    vlib::stepsic::write_toml "${CONFIG_DIR}/${config_name}.toml" \
        "GEOMETRY=cubical" \
        "LBOX=[${REF_LBOX_MPC_H}, ${REF_LBOX_MPC_H}, ${REF_LBOX_MPC_H}]" \
        "PERIODIC=[1, 1, 1]" \
        "R_3D=$(awk "BEGIN {print ${REF_LBOX_MPC_H} / 2.0}")" \
        "TYPE=grid" \
        "NGRID=${REF_NGRID}" \
        "IC_DIR=${output_dir}" \
        "H0=${COSMO_H0}" \
        "OMEGA_B=${COSMO_OMEGA_B}" \
        "OMEGA_M=${COSMO_OMEGA_M}" \
        "OMEGA_L=${COSMO_OMEGA_L}" \
        "SPECTRUM=${REF_SPECTRUM}" \
        "NONLINEAR=${REF_NONLINEAR}" \
        "HALOFIT=${REF_HALOFIT}" \
        "SEED=${REF_SEED}" \
        "LPTORDER=${lpt_order}" \
        "NMESH=${REF_NMESH}" \
        "REDSHIFT=${REF_Z_INIT}" \
        "INTERPOLATION=${REF_INTERPOLATION}" \
        "COMPENSATE=${REF_COMPENSATE}" \
        "SPHEREMODE=${REF_SPHEREMODE}" \
        "PAIRED=${REF_PAIRED}" \
        "NMESHSAMPLES=${REF_NMESHSAMPLES}" \
        "USE_DOUBLE=${REF_USE_DOUBLE}" \
        "SAVE_WHITE_NOISE=true"
    vlib::run_in_env "${STEPSIC_ENV}" \
        python "${STEPSIC_PY}" "${CONFIG_DIR}/${config_name}.toml"
}

_write_gadget_config
GADGET_BUILD_HASH="$(
    { sha256sum "${GADGET_CONFIG}"; printf '%s\n' "${GADGET_REVISION}"; } \
        | sha256sum | cut -c1-16
)"
GADGET_BUILD_STAGE="${BUILD_DIR}/${GADGET_BUILD_HASH}"
GADGET_BINARY="${GADGET_BUILD_STAGE}/Gadget4"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"
vlib::ensure_env "${GADGET4_ENV}"

echo ""
echo "========================================================================"
echo "  Periodic Gadget-4 reference"
echo "========================================================================"
echo "  Box: ${REF_LBOX_MPC_H} Mpc/h, grid: ${REF_NGRID}^3, PM: ${REF_PMGRID}^3"
echo "  z_init: ${REF_Z_INIT}, softening: ${REF_SOFTENING_MPC_H} Mpc/h"
echo "  Gadget revision: ${GADGET_REVISION}"
echo "  Run key: ${REFERENCE_RUN_KEY}"
echo "  Build key: ${GADGET_BUILD_HASH}"

if vlib::step_check "ic_load" "$(vlib::breadcrumb_get REF_IC_LOAD)"; then
    _write_stepsic_ic "${IC_LOAD_DIR}" 0 "ic-load"
    REF_IC_LOAD="$(vlib::find_ic "${IC_LOAD_DIR}")"
    vlib::breadcrumb_set "REF_IC_LOAD" "${REF_IC_LOAD}"
    vlib::step_done "ic_load"
fi
REF_IC_LOAD="${REF_IC_LOAD:-$(vlib::breadcrumb_get REF_IC_LOAD)}"

if vlib::step_check "ic_2lpt" "$(vlib::breadcrumb_get REF_IC_2LPT)"; then
    _write_stepsic_ic "${IC_2LPT_DIR}" 2 "ic-2lpt"
    REF_IC_2LPT="$(vlib::find_ic "${IC_2LPT_DIR}")"
    vlib::breadcrumb_set "REF_IC_2LPT" "${REF_IC_2LPT}"
    vlib::step_done "ic_2lpt"
fi
REF_IC_2LPT="${REF_IC_2LPT:-$(vlib::breadcrumb_get REF_IC_2LPT)}"

if vlib::step_check "ic_1lpt" "$(vlib::breadcrumb_get REF_IC_1LPT)"; then
    if [[ "${REF_RUN_1LPT}" == "true" ]]; then
        _write_stepsic_ic "${IC_1LPT_DIR}" 1 "ic-1lpt"
        REF_IC_1LPT="$(vlib::find_ic "${IC_1LPT_DIR}")"
        vlib::breadcrumb_set "REF_IC_1LPT" "${REF_IC_1LPT}"
    else
        REF_IC_1LPT=""
        vlib::breadcrumb_set "REF_IC_1LPT" ""
    fi
    vlib::step_done "ic_1lpt"
fi
REF_IC_1LPT="${REF_IC_1LPT:-$(vlib::breadcrumb_get REF_IC_1LPT)}"

if vlib::step_check "build" "${GADGET_BINARY}"; then
    mkdir -p "${GADGET_BUILD_STAGE}"
    cp "${GADGET_CONFIG}" "${GADGET_BUILD_STAGE}/Config.sh"
    vlib::run_in_env "${GADGET4_ENV}" bash -c '
        set -euo pipefail
        source_dir="$1"
        build_stage="$2"
        build_jobs="$3"
        exec make -C "${source_dir}" -j "${build_jobs}" \
            SYSTYPE=Generic-gcc DIR="${build_stage}" \
            CPP=mpicxx LINKER=mpicxx \
            "LIB_DIR=${CONDA_PREFIX}" \
            "HDF5_LIBS=-L${CONDA_PREFIX}/lib -lhdf5 ${CONDA_PREFIX}/lib/libz.so.1"
    ' _ "${GADGET4_SRC}" "${GADGET_BUILD_STAGE}" "${N_BUILD}"
    vlib::step_done "build"
fi

if vlib::step_check "run_2lpt" "$(vlib::breadcrumb_get REF_SNAP_2LPT)"; then
    [[ -f "${REF_IC_2LPT}" ]] || {
        echo "ERROR: 2LPT reference IC is missing." >&2
        exit 1
    }
    _write_output_list
    _write_gadget_param "${GADGET_PARAM_2LPT}" \
        "${REF_IC_2LPT}" "${RUN_2LPT_DIR}"
    vlib::clear_dir "${RUN_2LPT_DIR}"
    vlib::run_in_env "${GADGET4_ENV}" bash -c '
        cd "$1"
        OMP_NUM_THREADS="$2" exec mpirun -np "$3" "$4" "$5"
    ' _ "${GADGET_RUNTIME_DIR}" "${OMP_NUM_THREADS}" "${N_MPI}" \
        "${GADGET_BINARY}" "${GADGET_PARAM_2LPT}"
    REF_SNAP_2LPT="$(vlib::find_last_snap "${RUN_2LPT_DIR}")"
    [[ -n "${REF_SNAP_2LPT}" ]] || {
        echo "ERROR: Gadget-4 2LPT run produced no snapshot." >&2
        exit 1
    }
    vlib::breadcrumb_set "REF_SNAP_2LPT" "${REF_SNAP_2LPT}"
    vlib::step_done "run_2lpt"
fi
REF_SNAP_2LPT="${REF_SNAP_2LPT:-$(vlib::breadcrumb_get REF_SNAP_2LPT)}"

if vlib::step_check "run_1lpt" "$(vlib::breadcrumb_get REF_SNAP_1LPT)"; then
    if [[ "${REF_RUN_1LPT}" == "true" ]]; then
        [[ -f "${REF_IC_1LPT}" ]] || {
            echo "ERROR: 1LPT reference IC is missing." >&2
            exit 1
        }
        _write_output_list
        _write_gadget_param "${GADGET_PARAM_1LPT}" \
            "${REF_IC_1LPT}" "${RUN_1LPT_DIR}"
        vlib::clear_dir "${RUN_1LPT_DIR}"
        vlib::run_in_env "${GADGET4_ENV}" bash -c '
            cd "$1"
            OMP_NUM_THREADS="$2" exec mpirun -np "$3" "$4" "$5"
        ' _ "${GADGET_RUNTIME_DIR}" "${OMP_NUM_THREADS}" "${N_MPI}" \
            "${GADGET_BINARY}" "${GADGET_PARAM_1LPT}"
        REF_SNAP_1LPT="$(vlib::find_last_snap "${RUN_1LPT_DIR}")"
        [[ -n "${REF_SNAP_1LPT}" ]] || {
            echo "ERROR: Gadget-4 1LPT run produced no snapshot." >&2
            exit 1
        }
        vlib::breadcrumb_set "REF_SNAP_1LPT" "${REF_SNAP_1LPT}"
    fi
    vlib::step_done "run_1lpt"
fi
REF_SNAP_1LPT="${REF_SNAP_1LPT:-$(vlib::breadcrumb_get REF_SNAP_1LPT)}"

if vlib::step_check "publish" "${PAIR_MANIFEST}"; then
    REF_FIELD_DIR="$(dirname "${REF_IC_2LPT}")"
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/write-pair-manifest.py" \
        --snapshot "${REF_SNAP_2LPT}" \
        --ic "${REF_IC_2LPT}" \
        --load "${REF_IC_LOAD}" \
        --white-noise "${REF_FIELD_DIR}/ic_white_noise.hdf5" \
        --delta-k "${REF_FIELD_DIR}/ic_delta_k.hdf5" \
        --box-size-mpc-h "${REF_LBOX_MPC_H}" \
        --seed "${REF_SEED}" \
        --nmesh "${REF_NMESH}" \
        --ngrid "${REF_NGRID}" \
        --pm-grid "${REF_PMGRID}" \
        --lpt-order 2 \
        --initial-redshift "${REF_Z_INIT}" \
        --final-redshift "${REF_Z_FINAL}" \
        --H0-km-s-mpc "${COSMO_H0}" \
        --omega-b "${COSMO_OMEGA_B}" \
        --omega-m "${COSMO_OMEGA_M}" \
        --omega-lambda "${COSMO_OMEGA_L}" \
        --spectrum "${REF_SPECTRUM}" \
        --nonlinear "${REF_NONLINEAR}" \
        --halofit "${REF_HALOFIT}" \
        --interpolation "${REF_INTERPOLATION}" \
        --compensate "${REF_COMPENSATE}" \
        --sphere-mode "${REF_SPHEREMODE}" \
        --paired "${REF_PAIRED}" \
        --nmesh-samples "${REF_NMESHSAMPLES}" \
        --use-double "${REF_USE_DOUBLE}" \
        --softening-mpc-h "${REF_SOFTENING_MPC_H}" \
        --stepsic-revision "${STEPSIC_REVISION}" \
        --simulation-revision "${GADGET_REVISION}" \
        --output "${PAIR_MANIFEST}"
    vlib::step_done "publish"
fi

if vlib::step_check "compare" "${CONTROL_DATA}"; then
    if [[ "${REF_RUN_1LPT}" != "true" ]]; then
        echo "ERROR: periodic control requires REF_RUN_1LPT=true." >&2
        exit 1
    fi
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/measure-control.py" \
        --one-lpt "${REF_SNAP_1LPT}" \
        --two-lpt "${REF_SNAP_2LPT}" \
        --nmesh "${PK_NMESH}" \
        --kmax-frac-ny "${PK_KMAX_FRAC_NY}" \
        -o "${CONTROL_DATA}"
    vlib::step_done "compare"
fi

if vlib::step_check "plot" "${CONTROL_FIGURE}"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/plot-control.py" \
        -i "${CONTROL_DATA}" -o "${CONTROL_FIGURE}"
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/evaluate.py" \
        --archive "${CONTROL_DATA}" \
        --figure "${CONTROL_FIGURE}" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

CURRENT_LINK_TEMP="${REFERENCE_CACHE_ROOT}/.current-${REFERENCE_RUN_KEY}-$$"
ln -s "${REFERENCE_RUN_KEY}" "${CURRENT_LINK_TEMP}"
mv -Tf "${CURRENT_LINK_TEMP}" "${REFERENCE_CACHE_ROOT}/current"
vlib::report_done
