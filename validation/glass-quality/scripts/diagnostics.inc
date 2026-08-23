#!/usr/bin/env bash
# ============================================================================
#  Measure relaxed glasses and compare them with randomized particle loads.
#
#  For each geometry, the campaign measures power relative to shot noise,
#  nearest-neighbour spacing, radial density, and residual force when StePS
#  saved accelerations. It repeats each measurement after randomizing positions
#  while preserving particle masses and radial bands.
#
#  To measure an [L, L, L/s] torus with StePS's cubic periodic mode, the
#  campaign compresses z by 1/s and stacks s identical copies in a cube.
#
#  Pipeline steps:
#    1. twins        - randomize the available glasses
#    2. rescale      - compress and tile the cubic glass
#    3. build_force  - build StePS with SAVE_ACCELERATIONS
#    4. forces       - measure forces for every configuration
#    5. diagnose     - compute numerical diagnostics
#    6. plot         - plot glass quality and residual force
#
#  Steps 3-4 need the StePS toolchain. Without saved forces, steps 5-6 still
#  measure power, spacing, and density.
#
#  Usage:
#    bash run.sh [OPTIONS]
#
#  Options (standard vlib):
#    --size=SIZE / --step=N / --plot-only / --force / --force-step=A,B
#    --clean / --config=PATH / --list-steps / -h
#
#  Configuration: see the "Diagnostics" section of config.env.
# ============================================================================
set -euo pipefail

BASEDIR="${CAMPAIGN_DIR}"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign glass-quality "${BASEDIR}" "${BASEDIR}/config.env"

vlib::declare_steps twins rescale build_force forces diagnose plot evaluate
vlib::manifest_implementation twins "${BASEDIR}/scripts/twin.py"
vlib::manifest_implementation rescale "${BASEDIR}/scripts/rescale.py"
vlib::manifest_implementation diagnose "${BASEDIR}/scripts/diagnose.py"
vlib::manifest_implementation plot \
    "${BASEDIR}/scripts/plot-diagnostics.py" \
    "${BASEDIR}/scripts/plot-rescale.py"
vlib::manifest_evaluator evaluate \
    "${BASEDIR}/scripts/evaluate-diagnostics.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
for _snap_var in GLASS_SNAP_CUBICAL GLASS_SNAP_SPHERICAL GLASS_SNAP_CYLINDRICAL; do
    _snap="${!_snap_var}"
    if [[ -n "${_snap}" && ! -f "${_snap}" ]]; then
        echo "ERROR: ${_snap_var} not found: ${_snap}" >&2
        exit 1
    fi
done
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"
vlib::steps::validate_backend

export VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache/diagnostics"
OUTPUT="${VLIB_RUN_ROOT}/output"
FIGURES="${OUTPUT}/supplementary"
PARAM_DIR="${VLIB_RUN_ROOT}/config/diagnostics"
BUILD_DIR="${VLIB_RUN_ROOT}/build/diagnostics"
export PARAM_DIR BUILD_DIR
OUTDIR="${OUTDIR:-${VLIB_RUN_ROOT}/cache}"

TWIN_DIR="${VLIB_CACHE_DIR}/twins"
RESCALE_DIR="${VLIB_CACHE_DIR}/rescale"
FORCE_DIR="${VLIB_CACHE_DIR}/forces"
DIAG_DIR="${VLIB_CACHE_DIR}/diag"

if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
fi
mkdir -p "${TWIN_DIR}" "${RESCALE_DIR}" "${FORCE_DIR}" "${DIAG_DIR}" \
         "${FIGURES}" "${PARAM_DIR}" "${BUILD_DIR}"


vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

# -- Locate available glass snapshots ---------------------------------------
SNAP_CUBICAL="${GLASS_SNAP_CUBICAL:-$(vlib::find_last_snap "${OUTDIR}/cubic_random/glass")}"
SNAP_SPHERICAL="${GLASS_SNAP_SPHERICAL:-$(vlib::find_last_snap "${OUTDIR}/spherical/glass")}"
SNAP_CYLINDRICAL="${GLASS_SNAP_CYLINDRICAL:-$(vlib::find_last_snap "${OUTDIR}/cylindrical/glass")}"

for _snap in "${SNAP_CUBICAL}" "${SNAP_SPHERICAL}" "${SNAP_CYLINDRICAL}"; do
    if [[ -n "${_snap}" && -f "${_snap}" ]]; then
        for _step in twins forces diagnose; do
            vlib::manifest_input "${_step}" "${_snap}"
        done
    fi
done
if [[ -n "${SNAP_CUBICAL}" && -f "${SNAP_CUBICAL}" ]]; then
    vlib::manifest_input rescale "${SNAP_CUBICAL}"
fi

echo ""
echo "========================================================================"
echo "  Glass-quality diagnostics"
echo "========================================================================"
echo ""
echo "  cubical glass:     ${SNAP_CUBICAL:-<none>}"
echo "  spherical glass:   ${SNAP_SPHERICAL:-<none>}"
echo "  cylindrical glass: ${SNAP_CYLINDRICAL:-<none>}"
echo "  rescale aspects:   ${RESCALE_ASPECTS}"
echo ""

# geometry -> snapshot lookup used by several steps
_snap_for() {
    case "${1}" in
        cubical)     echo "${SNAP_CUBICAL}" ;;
        spherical)   echo "${SNAP_SPHERICAL}" ;;
        cylindrical) echo "${SNAP_CYLINDRICAL}" ;;
    esac
}

GEOMS=(cubical spherical cylindrical)

# -- Step 1: twins -----------------------------------------------------------
if vlib::step_check "twins"; then
    for geom in "${GEOMS[@]}"; do
        snap="$(_snap_for "${geom}")"
        [[ -z "${snap}" || ! -f "${snap}" ]] && {
            echo "  [skip] no ${geom} glass snapshot"; continue; }
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/twin.py" \
            "${snap}" "${TWIN_DIR}/twin_${geom}.hdf5" \
            --geometry "${geom}" --seed "${DIAG_TWIN_SEED}"
    done
    vlib::step_done "twins"
fi

# -- Step 2: rescale ---------------------------------------------------------
if vlib::step_check "rescale"; then
    if [[ -n "${SNAP_CUBICAL}" && -f "${SNAP_CUBICAL}" ]]; then
        for s in 1 ${RESCALE_ASPECTS}; do
            vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/rescale.py" \
                "${SNAP_CUBICAL}" "${RESCALE_DIR}/tiled_s${s}.hdf5" \
                --aspect "${s}"
        done
    else
        echo "  [skip] no cubical glass snapshot; rescale study unavailable"
    fi
    vlib::step_done "rescale"
fi

# Force-measurement binaries (built with SAVE_ACCELERATIONS; the IC is
# re-saved with the computed forces at run start).
STEPS_PRECISION_FLAGS="$(vlib::steps::precision_flags)"
STEPS_BACKEND_SUFFIX=""
STEPS_BACKEND_FLAGS=()
if [[ "${STEPS_BACKEND}" == "bh" ]]; then
    STEPS_BACKEND_SUFFIX="_bh"
    STEPS_BACKEND_FLAGS=("USE_BH=0.25" "RANDOMIZE_BH=123456")
fi
FORCE_BIN_PERIODIC="${BUILD_DIR}/StePS_force_periodic${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"
FORCE_BIN_SPHERICAL="${BUILD_DIR}/StePS_force_spherical${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"
FORCE_BIN_CYLINDRICAL="${BUILD_DIR}/StePS_force_cylindrical${STEPS_BACKEND_SUFFIX}$(vlib::steps::precision_suffix)"

# -- Step 3: build_force -----------------------------------------------------
if vlib::step_check "build_force" \
        "${FORCE_BIN_PERIODIC}" "${FORCE_BIN_SPHERICAL}" \
        "${FORCE_BIN_CYLINDRICAL}"; then
    vlib::ensure_env "${STEPS_ENV}"
    vlib::steps::detect_toolchain
    vlib::steps::build "$(basename "${FORCE_BIN_PERIODIC}")" \
        PERIODIC SAVE_ACCELERATIONS "${STEPS_BACKEND_FLAGS[@]}" \
        ${STEPS_PRECISION_FLAGS}
    vlib::steps::build "$(basename "${FORCE_BIN_SPHERICAL}")" \
        SAVE_ACCELERATIONS "${STEPS_BACKEND_FLAGS[@]}" \
        ${STEPS_PRECISION_FLAGS}
    vlib::steps::build "$(basename "${FORCE_BIN_CYLINDRICAL}")" \
        PERIODIC_Z SAVE_ACCELERATIONS "${STEPS_BACKEND_FLAGS[@]}" \
        ${STEPS_PRECISION_FLAGS}
    vlib::step_done "build_force"
fi

export EDS_H0
EDS_H0="$(vlib::cosmology::eds_h0 "${COSMO_H0}" "${COSMO_OMEGA_M}")"

# Run one force dump: _force_dump <name> <ic> <binary> <is_periodic>
#                                 <l_box> <r_sim> <softening> [ewald_table]
_force_dump() {
    local name="${1}" ic="${2}" binary="${3}" is_periodic="${4}"
    local l_box="${5}" r_sim="${6}" soft="${7}" ewald_table="${8:-}"
    local out="${FORCE_DIR}/${name}"
    local dump="${out}/initial_conditions_0000.hdf5"
    if [[ -f "${dump}" ]]; then
        echo "  [cached] force dump: ${name}"
        return 0
    fi
    mkdir -p "${out}"
    GLASS_A_START="${FORCE_A_START}" GLASS_A_MAX="${FORCE_A_MAX}" \
    GLASS_TIME_LIMIT_MIN="${FORCE_TIME_LIMIT_MIN}" \
    GLASS_FIRST_T_OUT=99999 GLASS_H_OUT=99999 \
        vlib::steps::write_glass_param \
            "force_${name}" "${ic}" "${out}/" \
            "${is_periodic}" "${l_box}" "${r_sim}" "${soft}" \
            "1000" "1000"
    if [[ -n "${ewald_table}" ]]; then
        vlib::steps::run_binary_with_fresh_ewald \
            "${binary}" "${PARAM_DIR}/force_${name}.param" \
            "${out}" "${ewald_table}"
    else
        vlib::steps::run_binary "${binary}" "${PARAM_DIR}/force_${name}.param"
    fi
    if [[ ! -f "${dump}" ]]; then
        echo "ERROR: force dump missing for ${name}: ${dump}" >&2
        return 1
    fi
}

# -- Step 4: forces ----------------------------------------------------------
if vlib::step_check "forces"; then
    vlib::ensure_env "${STEPS_ENV}"

    # Cubical glass + twin + rescaled tori (all in the periodic cube).
    if [[ -n "${SNAP_CUBICAL}" && -f "${SNAP_CUBICAL}" ]]; then
        for s in 1 ${RESCALE_ASPECTS}; do
            _force_dump "tiled_s${s}" "${RESCALE_DIR}/tiled_s${s}.hdf5" \
                "${FORCE_BIN_PERIODIC}" 3 "${LBOX}" 0 "${GLASS_SOFT_CUBIC}" \
                "Ewald_table_medres.hdf5"
        done
        _force_dump "twin_cubical" "${TWIN_DIR}/twin_cubical.hdf5" \
            "${FORCE_BIN_PERIODIC}" 3 "${LBOX}" 0 "${GLASS_SOFT_CUBIC}" \
            "Ewald_table_medres.hdf5"
    fi

    if [[ -n "${SNAP_SPHERICAL}" && -f "${SNAP_SPHERICAL}" ]]; then
        _force_dump "glass_spherical" "${SNAP_SPHERICAL}" \
            "${FORCE_BIN_SPHERICAL}" 0 0 "${R_3D}" "${GLASS_SOFT_SHELL}"
        _force_dump "twin_spherical" "${TWIN_DIR}/twin_spherical.hdf5" \
            "${FORCE_BIN_SPHERICAL}" 0 0 "${R_3D}" "${GLASS_SOFT_SHELL}"
    fi

    if [[ -n "${SNAP_CYLINDRICAL}" && -f "${SNAP_CYLINDRICAL}" ]]; then
        _force_dump "glass_cylindrical" "${SNAP_CYLINDRICAL}" \
            "${FORCE_BIN_CYLINDRICAL}" 4 "${LZ}" "${R_3D}" \
            "${GLASS_SOFT_SHELL}" "S1R2_Ewald_table_higres.hdf5"
        _force_dump "twin_cylindrical" "${TWIN_DIR}/twin_cylindrical.hdf5" \
            "${FORCE_BIN_CYLINDRICAL}" 4 "${LZ}" "${R_3D}" \
            "${GLASS_SOFT_SHELL}" "S1R2_Ewald_table_higres.hdf5"
    fi
    vlib::step_done "forces"
fi

# Prefer the force dump (has Accelerations) over the bare snapshot.
_diag_input() {
    local name="${1}" fallback="${2}"
    local dump="${FORCE_DIR}/${name}/initial_conditions_0000.hdf5"
    if [[ -f "${dump}" ]]; then echo "${dump}"; else echo "${fallback}"; fi
}

# -- Step 5: diagnose --------------------------------------------------------
if vlib::step_check "diagnose"; then
    DIAG_ARGS=(--nzones "${DIAG_NZONES}" --ncubes "${DIAG_NCUBES}"
               --pk-nmesh "${DIAG_PK_NMESH}" --nprof "${DIAG_NPROF}")
    for geom in "${GEOMS[@]}"; do
        snap="$(_snap_for "${geom}")"
        [[ -z "${snap}" || ! -f "${snap}" ]] && continue
        force_name="glass_${geom}"
        if [[ "${geom}" == "cubical" ]]; then
            force_name="tiled_s1"
        fi
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/diagnose.py" \
            "$(_diag_input "${force_name}" "${snap}")" \
            -o "${DIAG_DIR}/diag_${geom}.npz" \
            --geometry "${geom}" "${DIAG_ARGS[@]}"
        vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/diagnose.py" \
            "$(_diag_input "twin_${geom}" "${TWIN_DIR}/twin_${geom}.hdf5")" \
            -o "${DIAG_DIR}/diag_twin_${geom}.npz" \
            --geometry "${geom}" "${DIAG_ARGS[@]}"
    done
    # Rescaled tori: diagnose one tile (z < LBOX/s).
    if [[ -n "${SNAP_CUBICAL}" && -f "${SNAP_CUBICAL}" ]]; then
        for s in 1 ${RESCALE_ASPECTS}; do
            in="$(_diag_input "tiled_s${s}" "${RESCALE_DIR}/tiled_s${s}.hdf5")"
            [[ ! -f "${in}" ]] && continue
            tile_lz="$(awk "BEGIN {printf \"%.10f\", ${LBOX} / ${s}}")"
            vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/diagnose.py" \
                "${in}" -o "${DIAG_DIR}/diag_tiled_s${s}.npz" \
                --geometry cubical --tile-lz "${tile_lz}" "${DIAG_ARGS[@]}"
        done
    fi
    vlib::step_done "diagnose"
fi

# -- Step 6: plot ------------------------------------------------------------
if vlib::step_check "plot"; then
    for geom in "${GEOMS[@]}"; do
        g="${DIAG_DIR}/diag_${geom}.npz"
        t="${DIAG_DIR}/diag_twin_${geom}.npz"
        [[ ! -f "${g}" || ! -f "${t}" ]] && continue
        vlib::run_python "${STEPSIC_ENV}" \
            "${BASEDIR}/scripts/plot-diagnostics.py" \
            --glass "${g}" --twin "${t}" --label "${geom}" \
            -o "${FIGURES}/glass-quality-${geom}.pdf"
        echo "  -> ${FIGURES}/glass-quality-${geom}.pdf"
    done
    # Rescale summary needs force data on every point.
    RESCALE_PLOT_ARGS=()
    for s in 1 ${RESCALE_ASPECTS}; do
        d="${DIAG_DIR}/diag_tiled_s${s}.npz"
        [[ -f "${d}" ]] && RESCALE_PLOT_ARGS+=(--diag "${s}:${d}")
    done
    if [[ ${#RESCALE_PLOT_ARGS[@]} -ge 2 \
          && -f "${DIAG_DIR}/diag_twin_cubical.npz" ]]; then
        if vlib::run_python "${STEPSIC_ENV}" \
                "${BASEDIR}/scripts/plot-rescale.py" \
                "${RESCALE_PLOT_ARGS[@]}" \
                --twin "${DIAG_DIR}/diag_twin_cubical.npz" \
                -o "${FIGURES}/glass-rescale.pdf"; then
            echo "  -> ${FIGURES}/glass-rescale.pdf"
        else
            echo "  [skip] rescale summary (archives lack force data)"
        fi
    fi
    vlib::step_done "plot"
fi

if vlib::step_check "evaluate" "${OUTPUT}/result.json"; then
    EVALUATE_ARGS=()
    FIGURE_ARGS=()
    for geom in "${GEOMS[@]}"; do
        g="${DIAG_DIR}/diag_${geom}.npz"
        t="${DIAG_DIR}/diag_twin_${geom}.npz"
        if [[ -f "${g}" && -f "${t}" ]]; then
            EVALUATE_ARGS+=(--pair "${geom}" "${g}" "${t}")
            figure="${FIGURES}/glass-quality-${geom}.pdf"
            [[ -f "${figure}" ]] && FIGURE_ARGS+=(--figure "${figure}")
        fi
    done
    for s in 1 ${RESCALE_ASPECTS}; do
        archive="${DIAG_DIR}/diag_tiled_s${s}.npz"
        [[ -f "${archive}" ]] && EVALUATE_ARGS+=(--rescale "${s}" "${archive}")
    done
    [[ -f "${FIGURES}/glass-rescale.pdf" ]] && FIGURE_ARGS+=(--figure "${FIGURES}/glass-rescale.pdf")
    vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/evaluate-diagnostics.py" \
        "${EVALUATE_ARGS[@]}" "${FIGURE_ARGS[@]}" \
        --output "${OUTPUT}/result.json"
    vlib::step_done "evaluate"
fi

vlib::report_done
