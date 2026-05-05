#!/usr/bin/env bash
# ============================================================================
#  validation/glass/run.sh - Glass generation and validation figure controller
#
#  Top-level orchestrator for the glass validation pipeline. Configures
#  shared parameters, selectively dispatches geometry-specific pipelines,
#  and produces the final 4-panel validation figure.
#
#  Glass-making strategy (Gabor Racz's recipe):
#    All geometries use comoving integration in an Einstein-de Sitter (EdS)
#    background (Omega_m=1, Omega_lambda=0) with GLASS_MAKING enabled.
#    EdS avoids the Hubble-drag freeze-out that occurs in LCDM at late
#    times (a > 0.5), where Lambda dominates and structure formation stalls.
#    H0_EdS is fixed at 100 km/s/Mpc (h=1) so that StePS's rho_part /
#    rho_cosm check matches stepsic's h-independent IC mass density (which
#    is built around the canonical 100 km/s/Mpc); see vlib::cosmology::eds_h0
#    in _common/lib.sh. H0 is only a timescale here - comoving relaxation
#    is independent of it, so this does not affect the resulting glass.
#
#  Workflow (for each selected geometry):
#    1. Generate pre-glass ICs via stepsic (LPTORDER=0)
#    2. Write StePS parameter files
#    3. Compile StePS GPU binaries
#    4. Run StePS glass relaxation
#    Then: 5. Plot all available glass snapshots
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options (standard):
#    --step=N          Geometry sub-step to start from (1–4, default 1)
#    --plot-only       Skip all geometry pipelines; regenerate figure only
#    --force           Not used in glass (pass --no-clean to re-run)
#    --clean           Delete configs/, params/, builds/, and cache dirs
#    --config=PATH     Source an alternative config.env
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Options (glass-specific, pass as extra args):
#    --run=GEOM[,…]    Run only the listed geometries.
#                       Values: cubical, spherical, cylindrical, all
#                       (default: all)
#    --no-clean        Do not remove existing geometry outputs before running
#
#  GEOM_START_STEP env var:
#    Sets the intra-geometry start step (1=preglass, 2=param, 3=build, 4=run).
#    Equivalent to --step=N. GEOM_START_STEP takes precedence if set.
#
#  Examples:
#    # Full run, all geometries
#    N_GPU=4 bash run.sh
#
#    # Rerun only spherical from compilation, then replot everything
#    N_GPU=4 bash run.sh --run=spherical --step=3
#
#    # Just replot from existing snapshots
#    bash run.sh --plot-only
#
#  Configuration (edit config.env or export before running):
#    STEPS_SRC         Path to StePS source directory    (default: ~/projects/StePS)
#    STEPSIC_SRC       Path to stepsic source tree       (default: ~/projects/stepsic)
#    STEPS_ENV         Conda env for StePS               (default: steps)
#    STEPSIC_ENV       Conda env for stepsic             (default: stepsic)
#    N_MPI, N_GPU, OMP_NUM_THREADS
#    LBOX, NGRID, NPART, R_3D, D_4D, RCRIT, LZ
#    NRBINS_STEPS, NSHELL_STEPS, BIN_MODE_STEPS
#    NRBINS_CUBICAL, NSHELL_CUBICAL, BIN_MODE_CUBICAL
#    COSMOLOGY_NAME, COSMO_H0, COSMO_OMEGA_M, COSMO_OMEGA_L, COSMO_OMEGA_B
#    GLASS_TIME_LIMIT_MIN, GLASS_A_START, GLASS_A_MAX
#    GLASS_ACC_PARAM, GLASS_STEP_MIN, GLASS_STEP_MAX
#    GLASS_SOFT_SHELL, GLASS_SOFT_CUBIC
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::source_config "${VLIB_CONFIG_FILE:-${BASEDIR}/config.env}"

# Parse glass-specific flags from VLIB_EXTRA_ARGS
RUN_GEOMS="all"
DO_CLEAN="yes"

for arg in "${VLIB_EXTRA_ARGS[@]+"${VLIB_EXTRA_ARGS[@]}"}"; do
    case "${arg}" in
        --run=*)    RUN_GEOMS="${arg#--run=}" ;;
        --no-clean) DO_CLEAN="no" ;;
        *)
            echo "ERROR: Unknown argument '${arg}'." >&2
            echo "Usage: $0 [--run=GEOM,...] [--step=N] [--plot-only] [--no-clean] ..." >&2
            exit 1
            ;;
    esac
done

# GEOM_START_STEP: intra-geometry sub-step (1-4).
# --step=N and GEOM_START_STEP both set this; env var takes precedence.
export GEOM_START_STEP="${GEOM_START_STEP:-${VLIB_START_STEP}}"
if ! [[ "${GEOM_START_STEP}" =~ ^[1-4]$ ]]; then
    echo "ERROR: --step / GEOM_START_STEP must be in [1, 4] (got '${GEOM_START_STEP}')." >&2
    exit 1
fi

# Resolve which geometries to run.
RUN_CUBICAL="no"
RUN_SPHERICAL="no"
RUN_CYLINDRICAL="no"

if (( VLIB_PLOT_ONLY )); then
    : # leave all as "no"
elif [[ "${RUN_GEOMS}" == "all" ]]; then
    RUN_CUBICAL="yes"
    RUN_SPHERICAL="yes"
    RUN_CYLINDRICAL="yes"
else
    IFS=',' read -ra GEOM_LIST <<< "${RUN_GEOMS}"
    for g in "${GEOM_LIST[@]}"; do
        case "${g}" in
            cubical|cubic)        RUN_CUBICAL="yes" ;;
            spherical|sphere)     RUN_SPHERICAL="yes" ;;
            cylindrical|cylinder) RUN_CYLINDRICAL="yes" ;;
            none)                 : ;;
            *)
                echo "ERROR: Unknown geometry '${g}'. Use: cubical, spherical, cylindrical, all, none" >&2
                exit 1
                ;;
        esac
    done
fi

# Export shared configuration for geometry sub-scripts.
export STEPS_SRC STEPSIC_SRC STEPSIC_PY STEPS_ENV STEPSIC_ENV
export N_MPI N_GPU OMP_NUM_THREADS
export OMP_PLACES=cores OMP_PROC_BIND=close
export LBOX NGRID NPART R_3D D_4D RCRIT LZ
export NRBINS_STEPS NSHELL_STEPS BIN_MODE_STEPS
export NRBINS_CUBICAL NSHELL_CUBICAL BIN_MODE_CUBICAL
export COSMOLOGY_NAME COSMO_OMEGA_B COSMO_OMEGA_M COSMO_OMEGA_L COSMO_H0
export GLASS_TIME_LIMIT_MIN GLASS_A_START GLASS_A_MAX
export GLASS_ACC_PARAM GLASS_STEP_MIN GLASS_STEP_MAX
export GLASS_SOFT_SHELL GLASS_SOFT_CUBIC
export GLASS_FIRST_T_OUT GLASS_H_OUT
[[ -n "${CXX:-}" ]]       && export CXX
[[ -n "${CUDA_PATH:-}" ]] && export CUDA_PATH
[[ -n "${MPI_INC:-}" ]]   && export MPI_INC
[[ -n "${MPI_LIBS:-}" ]]  && export MPI_LIBS
[[ -n "${HDF5_INC:-}" ]]  && export HDF5_INC
[[ -n "${HDF5_LIBS:-}" ]] && export HDF5_LIBS

# Directory layout (also exported for geometry sub-scripts).
export OUTDIR="${OUTDIR:-${BASEDIR}}"
export TOML_DIR="${OUTDIR}/configs"
export PARAM_DIR="${OUTDIR}/params"
export BUILD_DIR="${OUTDIR}/builds"
OUTPUT="${BASEDIR}/output"

if (( VLIB_LIST_STEPS )); then
    echo "  Geometry sub-steps (1-4, controlled by --step=N or GEOM_START_STEP):"
    echo "     1. preglass    - generate pre-glass ICs via stepsic"
    echo "     2. param       - write StePS parameter files"
    echo "     3. build       - compile StePS GPU binary"
    echo "     4. run         - run StePS glass relaxation"
    echo ""
    echo "  Controller steps:"
    echo "     cubical     - run cubical (T^3) geometry pipeline"
    echo "     spherical   - run spherical (R^3) geometry pipeline"
    echo "     cylindrical - run cylindrical (S^1×R^2) geometry pipeline"
    echo "     plot        - generate 4-panel validation figure"
    exit 0
fi

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"
if [[ "${RUN_CUBICAL}" == "yes" || "${RUN_SPHERICAL}" == "yes" || "${RUN_CYLINDRICAL}" == "yes" ]]; then
    vlib::ensure_env "${STEPS_ENV}"
fi

# Compute EdS-equivalent H0 and export for geometry sub-scripts.
export EDS_H0
EDS_H0="$(vlib::cosmology::eds_h0 "${COSMO_H0}" "${COSMO_OMEGA_M}")"

mkdir -p "${TOML_DIR}" "${PARAM_DIR}" "${BUILD_DIR}" "${OUTPUT}"

echo ""
echo "========================================================================"
echo "  Glass validation pipeline"
echo "========================================================================"
echo ""
echo "  Target cosmology:  H0=${COSMO_H0}, Omega_m=${COSMO_OMEGA_M}"
echo "  Glass-making EdS:  H0_EdS=${EDS_H0}, Omega_m=1.0"
echo "  Scale factor:      a_start=${GLASS_A_START} → a_max=${GLASS_A_MAX}"
echo "  Geometries:        cubical=${RUN_CUBICAL}  spherical=${RUN_SPHERICAL}  cylindrical=${RUN_CYLINDRICAL}"
echo "  Geom start step:   ${GEOM_START_STEP}"
echo ""

# -- Selective cleanup ------------------------------------------------------
if [[ "${DO_CLEAN}" == "yes" && ! (( VLIB_PLOT_ONLY )) ]]; then
    echo "------------------------------------------------------------------------"
    echo "  Cleaning selected geometry outputs"
    echo "------------------------------------------------------------------------"

    if [[ "${RUN_CUBICAL}" == "yes" ]]; then
        echo "  Cleaning cubical..."
        vlib::clear_files "${TOML_DIR}"  "cubic_random.toml" "cubic_grid.toml"
        vlib::clear_files "${PARAM_DIR}" "cubic_random.param"
        vlib::clear_files "${BUILD_DIR}" "StePS_glass_periodic"
        vlib::clear_dir "${OUTDIR}/cubic_random/preglass"
        vlib::clear_dir "${OUTDIR}/cubic_random/glass"
        vlib::clear_dir "${OUTDIR}/cubic_grid/preglass"
    fi

    if [[ "${RUN_SPHERICAL}" == "yes" ]]; then
        echo "  Cleaning spherical..."
        vlib::clear_files "${TOML_DIR}"  "spherical.toml"
        vlib::clear_files "${PARAM_DIR}" "spherical.param"
        vlib::clear_files "${BUILD_DIR}" "StePS_glass_spherical"
        vlib::clear_dir "${OUTDIR}/spherical/preglass"
        vlib::clear_dir "${OUTDIR}/spherical/glass"
    fi

    if [[ "${RUN_CYLINDRICAL}" == "yes" ]]; then
        echo "  Cleaning cylindrical..."
        vlib::clear_files "${TOML_DIR}"  "cylindrical.toml"
        vlib::clear_files "${PARAM_DIR}" "cylindrical.param"
        vlib::clear_files "${BUILD_DIR}" "StePS_glass_cylindrical"
        vlib::clear_dir "${OUTDIR}/cylindrical/preglass"
        vlib::clear_dir "${OUTDIR}/cylindrical/glass"
    fi
fi

# -- Dispatch geometry pipelines --------------------------------------------
[[ "${RUN_CUBICAL}"     == "yes" ]] && bash "${BASEDIR}/geometries/cubical.sh"
[[ "${RUN_SPHERICAL}"   == "yes" ]] && bash "${BASEDIR}/geometries/spherical.sh"
[[ "${RUN_CYLINDRICAL}" == "yes" ]] && bash "${BASEDIR}/geometries/cylindrical.sh"

# -- Plot step --------------------------------------------------------------
echo ""
echo "------------------------------------------------------------------------"
echo "  Generating validation figure"
echo "------------------------------------------------------------------------"

SNAP_CUBIC_RANDOM="$(vlib::find_last_snap "${OUTDIR}/cubic_random/glass")"
SNAP_CUBIC_GRID="$(vlib::find_ic "${OUTDIR}/cubic_grid/preglass" 2>/dev/null || true)"
SNAP_SPHERICAL="$(vlib::find_last_snap "${OUTDIR}/spherical/glass")"
SNAP_CYLINDRICAL="$(vlib::find_last_snap "${OUTDIR}/cylindrical/glass")"

PLOT_ARGS=()
if [[ -n "${SNAP_CUBIC_RANDOM}" && -f "${SNAP_CUBIC_RANDOM}" ]]; then
    PLOT_ARGS+=(--cubic-random "${SNAP_CUBIC_RANDOM}")
else
    echo "  WARNING: No cubic random glass snapshot found; panel will be blank."
fi
if [[ -n "${SNAP_CUBIC_GRID}" && -f "${SNAP_CUBIC_GRID}" ]]; then
    PLOT_ARGS+=(--cubic-grid "${SNAP_CUBIC_GRID}")
else
    echo "  WARNING: No cubic grid IC found; panel will be blank."
fi
if [[ -n "${SNAP_SPHERICAL}" && -f "${SNAP_SPHERICAL}" ]]; then
    PLOT_ARGS+=(--spherical "${SNAP_SPHERICAL}")
else
    echo "  WARNING: No spherical glass snapshot found; panel will be blank."
fi
if [[ -n "${SNAP_CYLINDRICAL}" && -f "${SNAP_CYLINDRICAL}" ]]; then
    PLOT_ARGS+=(--cylindrical "${SNAP_CYLINDRICAL}")
else
    echo "  WARNING: No cylindrical glass snapshot found; panel will be blank."
fi

if [[ ${#PLOT_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No snapshots found for any geometry. Nothing to plot." >&2
    exit 1
fi

PLOT_ARGS+=(
    --target-radius "${R_3D}"
    --slice-thickness 100
    --fraction 1.0
    -o "${OUTPUT}/glass.pdf"
)

vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" "${PLOT_ARGS[@]}"

echo "  -> ${OUTPUT}/glass.pdf"

vlib::report_done
