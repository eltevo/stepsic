#!/usr/bin/env bash
# ============================================================================
#  Measure how Fourier-box padding affects a non-periodic domain.
#
#  The campaign varies alpha = LBOX/(2*R_3D) while keeping voxel size and
#  white noise inside the domain fixed. It compares each box with the largest:
#    - displacement/velocity error in radial shells,
#    - correlation between opposite sides of the boundary shell,
#    - central-region mass-weighted P(k),
#    - mass remaining inside the domain and radial motion at its boundary.
#  The reported values combine NSEEDS random fields.
#
#  Steps:
#    1. run               padding sweep for GEOMETRY at NGRID0
#    2. comparison-runs   the same sweep for the other geometry and at NGRID0/2
#    3. plot              sweep figure and geometry/mesh comparison figure
#    4. controls          1LPT of a Gaussian dipole against its exact
#                         open-domain displacement
#    5. cosmological      linear structure functions of padded boxes against
#                         open space, for both geometries
#    6. evaluate          padding convergence check and recorded metrics
#
#  With both RELATIVE_MOTION_*_GLASS paths set, eight further steps use the
#  relaxed glasses of the spherical and cylindrical domains:
#    7. glass_prepare        embedded mesh geometry and seeds of both glasses
#    8. glass_run            particle displacements of every padded box and seed
#    9. glass_measure        change of the RMS relative displacement with padding
#   10. two_point            relative displacement of two points in each box and
#                            in the infinite universe, from sums over the modes
#   11. plot_relative_motion measured changes against the sums over modes
#   12. monopole             mean density contrast, mean strain and opposite
#                            boundary points of the open domain, 2LPT ensemble
#   13. plot_two_point       article figure embedding-two-point.pdf
#   14. mixing_variance      modes shared between adjacent multiresolution levels
#
#  With EVOLVED_PAIR_DIR, EVOLVED_PAIR_RANDOMS and EVOLVED_PAIR_GLASS set, two
#  further steps compare two evolved cylinders, one computed in a box of one
#  diameter and one in a box of two diameters:
#   15. evolved_pair         power spectra, displacement differences, centre of
#                            mass and mean density of the pair at z = 0
#   16. plot_evolved_pair    article figure evolved-pair.pdf
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options:
#    --size=SIZE       small, medium (default), or approved large profile
#    --step=N          Start from step N (default 1)
#    --plot-only       Skip the measurement steps; redraw the plot step's PDFs
#    --force           Re-run every step, ignoring cached outputs
#    --force-step=A,B  Re-run only the named steps
#    --clean           Delete cache/ and output/ before running
#    --config=PATH     Select a typed custom TOML profile
#    --list-steps      Print the step list and exit
#    -h, --help        Print this help text and exit
#
#  Configuration (edit config.env or export before running):
#    GEOMETRY          spherical | cylindrical       (default: spherical)
#    R_3D, D_4D, RCRIT, BIN_MODE, NRBINS, NSHELL, LZ
#    NGRID0            Voxels across 2*R_3D          (default: 128)
#    ALPHAS            Padding ratios, ascending     (default: 1.0 .. 2.0)
#    LPT, Z, SEED, NSEEDS, LOAD_SEED
#    BOUNDARY_FRAC, CORE_FRAC, PK_NMESH, NPROF
#    FILTER_SCALE      Smoothing wavenumber of the cosmological step [h/Mpc]
#    RELATIVE_MOTION_{SPHERICAL,CYLINDRICAL}_{GLASS,RADIUS,CELLS},
#    RELATIVE_MOTION_CYLINDRICAL_PERIOD, RELATIVE_MOTION_NSEEDS,
#    TWO_POINT_{SPHERICAL,CYLINDRICAL}_CELLS, TWO_POINT_PADDINGS,
#    TWO_POINT_FILTER_SCALE, MONOPOLE_{SPHERICAL,CYLINDRICAL}_CELLS,
#    MONOPOLE_NSEEDS, MIXING_NSEEDS
#    EVOLVED_PAIR_{DIR,RANDOMS,GLASS,IC,RADIUS,OUTER,BINS,NMESH,P0,KMAX}, STEPS_SRC
#    STEPSIC_ENV       Conda env for stepsic         (default: stepsic)
# ============================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign periodic-embedding "${BASEDIR}" "${BASEDIR}/config.env"

EMBEDDING_STEPS=(run comparison-runs plot controls cosmological evaluate)
RELATIVE_MOTION=0
if [[ -n "${RELATIVE_MOTION_SPHERICAL_GLASS}" && -n "${RELATIVE_MOTION_CYLINDRICAL_GLASS}" ]]; then
    RELATIVE_MOTION=1
    EMBEDDING_STEPS+=(glass_prepare glass_run glass_measure two_point
        plot_relative_motion monopole plot_two_point mixing_variance)
fi
EVOLVED_PAIR=0
if [[ -n "${EVOLVED_PAIR_DIR}" && -n "${EVOLVED_PAIR_RANDOMS}" && -n "${EVOLVED_PAIR_GLASS}" ]]; then
    EVOLVED_PAIR=1
fi
EMBEDDING_STEPS+=(evolved_pair plot_evolved_pair)
vlib::declare_steps "${EMBEDDING_STEPS[@]}"
vlib::manifest_implementation run "${BASEDIR}/scripts/run.py"
vlib::manifest_implementation comparison-runs "${BASEDIR}/scripts/run.py"
vlib::manifest_implementation plot "${BASEDIR}/scripts/plot.py"
vlib::manifest_implementation controls "${BASEDIR}/scripts/controls.py"
vlib::manifest_implementation cosmological "${BASEDIR}/scripts/cosmological.py"
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
for step in glass_prepare glass_run glass_measure; do
    vlib::manifest_implementation "${step}" "${BASEDIR}/scripts/relative-motion.py"
done
vlib::manifest_implementation two_point "${BASEDIR}/scripts/two-point.py"
vlib::manifest_implementation plot_relative_motion "${BASEDIR}/scripts/plot-relative-motion.py"
vlib::manifest_implementation monopole "${BASEDIR}/scripts/monopole.py"
vlib::manifest_implementation plot_two_point "${BASEDIR}/scripts/plot-two-point.py"
vlib::manifest_implementation mixing_variance "${BASEDIR}/scripts/mixing-variance.py"
vlib::manifest_implementation evolved_pair "${BASEDIR}/scripts/evolved-pair.py" \
    "${STEPS_SRC}/tools/PowerSpectra/StePS_Pk.py"
vlib::manifest_implementation plot_evolved_pair "${BASEDIR}/scripts/plot-evolved-pair.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"
if (( RELATIVE_MOTION )); then
    vlib::manifest_input glass_prepare \
        "${RELATIVE_MOTION_SPHERICAL_GLASS}" "${RELATIVE_MOTION_CYLINDRICAL_GLASS}"
fi
# The snapshots are large; the listing of their names and sizes stands for all
# of them, and the initial conditions, the last snapshots, the randoms and the
# glass are hashed in full.
PAIR_LISTING="${VLIB_RUN_ROOT}/cache/evolved-pair-snapshots.txt"
if (( EVOLVED_PAIR )); then
    vlib::manifest_input evolved_pair "${PAIR_LISTING}" \
        "${EVOLVED_PAIR_DIR}/ics/a/${EVOLVED_PAIR_IC}" "${EVOLVED_PAIR_DIR}/ics/b/${EVOLVED_PAIR_IC}" \
        "${EVOLVED_PAIR_DIR}/sim_a/snapshot_0009.hdf5" "${EVOLVED_PAIR_DIR}/sim_b/snapshot_0009.hdf5" \
        "${EVOLVED_PAIR_RANDOMS}" "${EVOLVED_PAIR_GLASS}"
fi

VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache"
OUTPUT="${VLIB_RUN_ROOT}/output"
FIGURES="${OUTPUT}/supplementary"

# --clean: wipe cache and output before running
if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi

mkdir -p "${VLIB_CACHE_DIR}" "${FIGURES}"

if [[ "${GEOMETRY}" == "spherical" ]]; then
    OTHER_GEOMETRY="cylindrical"
else
    OTHER_GEOMETRY="spherical"
fi
HALF_NGRID0=$(( NGRID0 / 2 ))
COMPARISON_ARCHIVES=(
    "${VLIB_CACHE_DIR}/${GEOMETRY}-${HALF_NGRID0}.npz"
    "${VLIB_CACHE_DIR}/${OTHER_GEOMETRY}-${NGRID0}.npz"
    "${VLIB_CACHE_DIR}/${OTHER_GEOMETRY}-${HALF_NGRID0}.npz"
)

# run_sweep GEOMETRY NGRID0 OUTPUT
run_sweep() {
    # shellcheck disable=SC2086
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/run.py" \
        --geometry "${1}" \
        --R3d "${R_3D}" \
        --D4d "${D_4D}" \
        --rcrit "${RCRIT}" \
        --bin-mode "${BIN_MODE}" \
        --nrbins "${NRBINS}" \
        --nshell "${NSHELL}" \
        --Lz "${LZ}" \
        --ngrid0 "${2}" \
        --alphas ${ALPHAS} \
        --lpt "${LPT}" \
        --z "${Z}" \
        --seed "${SEED}" \
        --nseeds "${NSEEDS}" \
        --load-seed "${LOAD_SEED}" \
        --boundary-frac "${BOUNDARY_FRAC}" \
        --core-frac "${CORE_FRAC}" \
        --pk-nmesh "${PK_NMESH}" \
        --nprof "${NPROF}" \
        -o "${3}"
}

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

echo ""
echo "========================================================================"
echo "  Padding validation: periodic-embedding error vs alpha"
echo "========================================================================"
echo ""
echo "  geometry:    ${GEOMETRY}   R_3D: ${R_3D} Mpc/h   RCRIT: ${RCRIT}"
echo "  ngrid0:      ${NGRID0}   alphas: ${ALPHAS}"
echo "  lpt:         ${LPT}   z: ${Z}   seeds: ${SEED}+${NSEEDS}"
echo "  cache:       ${VLIB_CACHE_DIR}"
echo "  output:      ${FIGURES}/periodic-embedding.pdf"
echo ""

# --------------------------------------------------------------------------
if vlib::step_check "run" "${VLIB_CACHE_DIR}/data.npz"; then
    run_sweep "${GEOMETRY}" "${NGRID0}" "${VLIB_CACHE_DIR}/data.npz"
    vlib::step_done "run"
fi

if vlib::step_check "comparison-runs" "${COMPARISON_ARCHIVES[@]}"; then
    run_sweep "${GEOMETRY}" "${HALF_NGRID0}" "${COMPARISON_ARCHIVES[0]}"
    run_sweep "${OTHER_GEOMETRY}" "${NGRID0}" "${COMPARISON_ARCHIVES[1]}"
    run_sweep "${OTHER_GEOMETRY}" "${HALF_NGRID0}" "${COMPARISON_ARCHIVES[2]}"
    vlib::step_done "comparison-runs"
fi

# --------------------------------------------------------------------------
if vlib::step_check "plot" "${FIGURES}/periodic-embedding.pdf" \
    "${FIGURES}/padding-comparison.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot.py" \
        -i "${VLIB_CACHE_DIR}/data.npz" \
        -o "${FIGURES}/periodic-embedding.pdf" \
        --comparison "${COMPARISON_ARCHIVES[@]}" \
        --comparison-output "${FIGURES}/padding-comparison.pdf"
    vlib::step_done "plot"
fi

if vlib::step_check controls "${VLIB_CACHE_DIR}/controls.npz" "${FIGURES}/embedding-controls.pdf"; then
    # shellcheck disable=SC2086
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/controls.py" \
        --nmesh "${NGRID0}" --geometry "${GEOMETRY}" --radius "${R_3D}" \
        --length "${LZ}" --alphas ${ALPHAS} --seed "${SEED}" \
        --output "${VLIB_CACHE_DIR}/controls.npz" --figure "${FIGURES}/embedding-controls.pdf"
    vlib::step_done controls
fi

if vlib::step_check cosmological "${VLIB_CACHE_DIR}/cosmological.npz" "${FIGURES}/cosmological-comparison.pdf"; then
    # shellcheck disable=SC2086
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/cosmological.py" \
        --nmesh "${NGRID0}" --radius "${R_3D}" --length "${LZ}" \
        --redshift "${Z}" --filter-scale "${FILTER_SCALE}" --alphas ${ALPHAS} \
        --output "${VLIB_CACHE_DIR}/cosmological.npz" --figure "${FIGURES}/cosmological-comparison.pdf"
    vlib::step_done cosmological
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --archive "${VLIB_CACHE_DIR}/data.npz" \
        --controls "${VLIB_CACHE_DIR}/controls.npz" \
        --figure "${FIGURES}/periodic-embedding.pdf" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

if (( RELATIVE_MOTION )); then
    WORK="${VLIB_CACHE_DIR}/relative-motion"

    if vlib::step_check glass_prepare "${WORK}/inputs.npz"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/relative-motion.py" prepare \
            --work "${WORK}" \
            --spherical-glass "${RELATIVE_MOTION_SPHERICAL_GLASS}" \
            --spherical-radius-mpc "${RELATIVE_MOTION_SPHERICAL_RADIUS}" \
            --spherical-cells "${RELATIVE_MOTION_SPHERICAL_CELLS}" \
            --cylindrical-glass "${RELATIVE_MOTION_CYLINDRICAL_GLASS}" \
            --cylindrical-radius-mpc "${RELATIVE_MOTION_CYLINDRICAL_RADIUS}" \
            --cylindrical-period-mpc "${RELATIVE_MOTION_CYLINDRICAL_PERIOD}" \
            --cylindrical-cells "${RELATIVE_MOTION_CYLINDRICAL_CELLS}" \
            --seeds "${RELATIVE_MOTION_NSEEDS}"
        vlib::step_done glass_prepare
    fi

    # The particle archives are large; the listing of their names and sizes
    # stands for them in the manifest.
    if vlib::step_check glass_run "${WORK}/particles.txt"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/relative-motion.py" \
            run --work "${WORK}"
        find "${WORK}/particles" -name '*.npz' -printf '%f %s\n' | sort > "${WORK}/particles.txt"
        vlib::step_done glass_run
    fi

    if vlib::step_check glass_measure "${WORK}/relative-motion.npz"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/relative-motion.py" \
            measure --work "${WORK}"
        vlib::step_done glass_measure
    fi

    if vlib::step_check two_point "${VLIB_CACHE_DIR}/two-point.npz"; then
        # shellcheck disable=SC2086
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/two-point.py" \
            --spherical-radius-mpc "${RELATIVE_MOTION_SPHERICAL_RADIUS}" \
            --spherical-cells "${TWO_POINT_SPHERICAL_CELLS}" \
            --cylindrical-radius-mpc "${RELATIVE_MOTION_CYLINDRICAL_RADIUS}" \
            --cylindrical-period-mpc "${RELATIVE_MOTION_CYLINDRICAL_PERIOD}" \
            --cylindrical-cells "${TWO_POINT_CYLINDRICAL_CELLS}" \
            --paddings ${TWO_POINT_PADDINGS} \
            --filter-scale "${TWO_POINT_FILTER_SCALE}" \
            --glass-work "${WORK}" \
            --output "${VLIB_CACHE_DIR}/two-point.npz"
        vlib::step_done two_point
    fi

    if vlib::step_check plot_relative_motion "${FIGURES}/embedding-relative-motion.pdf"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-relative-motion.py" \
            --measurement "${WORK}/relative-motion.npz" \
            --prediction "${VLIB_CACHE_DIR}/two-point.npz" \
            --output "${FIGURES}/embedding-relative-motion.pdf"
        vlib::step_done plot_relative_motion
    fi

    if vlib::step_check monopole "${VLIB_CACHE_DIR}/monopole.npz"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/monopole.py" \
            --spherical-radius-mpc "${RELATIVE_MOTION_SPHERICAL_RADIUS}" \
            --spherical-cells "${MONOPOLE_SPHERICAL_CELLS}" \
            --cylindrical-radius-mpc "${RELATIVE_MOTION_CYLINDRICAL_RADIUS}" \
            --cylindrical-period-mpc "${RELATIVE_MOTION_CYLINDRICAL_PERIOD}" \
            --cylindrical-cells "${MONOPOLE_CYLINDRICAL_CELLS}" \
            --paddings 1 2 3 \
            --seeds "${MONOPOLE_NSEEDS}" \
            --output "${VLIB_CACHE_DIR}/monopole.npz"
        vlib::step_done monopole
    fi

    if vlib::step_check plot_two_point "${OUTPUT}/embedding-two-point.pdf"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-two-point.py" \
            --two-point "${VLIB_CACHE_DIR}/two-point.npz" \
            --ensemble "${VLIB_CACHE_DIR}/monopole.npz" \
            --output "${OUTPUT}/embedding-two-point.pdf"
        vlib::step_done plot_two_point
    fi

    if vlib::step_check mixing_variance "${VLIB_CACHE_DIR}/mixing-variance.npz"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/mixing-variance.py" \
            --inputs "${WORK}/inputs.npz" \
            --seeds "${MIXING_NSEEDS}" \
            --output "${VLIB_CACHE_DIR}/mixing-variance.npz"
        vlib::step_done mixing_variance
    fi
fi

if (( EVOLVED_PAIR )); then
    mkdir -p "${VLIB_CACHE_DIR}"
    find "${EVOLVED_PAIR_DIR}" -name 'snapshot_*.hdf5' -printf '%P %s\n' | sort > "${PAIR_LISTING}"

    if vlib::step_check evolved_pair "${VLIB_CACHE_DIR}/evolved-pair.npz"; then
        # shellcheck disable=SC2086
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evolved-pair.py" \
            --pair-dir "${EVOLVED_PAIR_DIR}" \
            --ic "${EVOLVED_PAIR_IC}" \
            --randoms "${EVOLVED_PAIR_RANDOMS}" \
            --glass "${EVOLVED_PAIR_GLASS}" \
            --steps-src "${STEPS_SRC}" \
            --domain-radius "${RELATIVE_MOTION_CYLINDRICAL_RADIUS}" \
            --radius "${EVOLVED_PAIR_RADIUS}" \
            --outer ${EVOLVED_PAIR_OUTER} \
            --bins "${EVOLVED_PAIR_BINS}" \
            --nmesh "${EVOLVED_PAIR_NMESH}" \
            --p0 "${EVOLVED_PAIR_P0}" \
            --k-max "${EVOLVED_PAIR_KMAX}" \
            --output "${VLIB_CACHE_DIR}/evolved-pair.npz"
        vlib::step_done evolved_pair
    fi

    if vlib::step_check plot_evolved_pair "${OUTPUT}/evolved-pair.pdf"; then
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/plot-evolved-pair.py" \
            --input "${VLIB_CACHE_DIR}/evolved-pair.npz" \
            --output "${OUTPUT}/evolved-pair.pdf"
        vlib::step_done plot_evolved_pair
    fi
else
    vlib::step_skip evolved_pair "EVOLVED_PAIR_DIR, EVOLVED_PAIR_RANDOMS or EVOLVED_PAIR_GLASS not set"
    vlib::step_skip plot_evolved_pair "EVOLVED_PAIR_DIR, EVOLVED_PAIR_RANDOMS or EVOLVED_PAIR_GLASS not set"
fi

vlib::report_done
