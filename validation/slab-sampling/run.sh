#!/usr/bin/env bash
# ==========================================================================
# Compare slab anisotropy, power transfer, and a matching cut from a cube.
#
# Usage: bash run.sh [--size=small|medium|large] [standard campaign options]
# Primary figures: output/slab-transfer.pdf and output/slab-fairness.pdf
# Supplementary figure: output/supplementary/slab-anisotropy.pdf
# ==========================================================================
set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BASEDIR}/../_common/lib.sh"

vlib::parse_args "$@"
vlib::prepare_campaign slab-sampling "${BASEDIR}" "${BASEDIR}/config.env"
vlib::declare_steps \
    field_histogram transfer fairness plot_field plot_transfer plot_fairness \
    evaluate_transfer evaluate_fairness evaluate
vlib::manifest_implementation field_histogram "${BASEDIR}/scripts/field/run.py"
vlib::manifest_implementation transfer "${BASEDIR}/scripts/transfer/run.py"
vlib::manifest_implementation fairness "${BASEDIR}/scripts/fairness/run.py"
vlib::manifest_implementation plot_field "${BASEDIR}/scripts/field/plot.py"
vlib::manifest_implementation plot_transfer \
    "${BASEDIR}/scripts/transfer/plot.py" "${BASEDIR}/scripts/transfer/common.py"
vlib::manifest_implementation plot_fairness "${BASEDIR}/scripts/fairness/plot.py"
vlib::manifest_evaluator evaluate_transfer "${BASEDIR}/scripts/transfer/evaluate.py"
vlib::manifest_implementation evaluate_transfer "${BASEDIR}/scripts/transfer/common.py"
vlib::manifest_evaluator evaluate_fairness "${BASEDIR}/scripts/fairness/evaluate.py"
vlib::manifest_evaluator evaluate "${BASEDIR}/scripts/evaluate.py"
if (( VLIB_LIST_STEPS )); then
    vlib::profile_summary
    vlib::list_steps
    exit 0
fi
vlib::manifest_init "${BASEDIR}" "${CONFIG_FILE}"

VLIB_CACHE_DIR="${VLIB_RUN_ROOT}/cache"
OUTPUT="${VLIB_RUN_ROOT}/output"
SUPPLEMENTARY="${OUTPUT}/supplementary"
if (( VLIB_CLEAN )); then
    vlib::clear_dir "${VLIB_CACHE_DIR}"
    vlib::clear_dir "${OUTPUT}"
fi
mkdir -p "${VLIB_CACHE_DIR}" "${OUTPUT}" "${SUPPLEMENTARY}"

vlib::init_conda
vlib::ensure_env "${STEPSIC_ENV}"

TRANSFER_PAIRED_FLAG=()
[[ "${TRANSFER_PAIRED}" == true ]] && TRANSFER_PAIRED_FLAG=(--paired)
FAIR_PAIRED_FLAG=()
[[ "${FAIR_PAIRED}" == true ]] && FAIR_PAIRED_FLAG=(--paired)

# shellcheck disable=SC2086
if vlib::step_check field_histogram "${VLIB_CACHE_DIR}/field.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/field/run.py" \
        --Lbox ${LBOX_SLAB} --nmesh "${FIELD_NMESH}" --lpt 2 --z "${Z}" \
        --method "${METHOD}" --seed "${SEED}" --nreal "${FIELD_NREAL}" \
        --nbins "${FIELD_NBINS}" -o "${VLIB_CACHE_DIR}/field.npz"
    vlib::step_done field_histogram
fi

if vlib::step_check transfer "${VLIB_CACHE_DIR}/transfer.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/transfer/run.py" \
        --Lcube "${LCUBE}" --Lz-min "${LZ_MIN}" --nsteps "${TRANSFER_NSTEPS}" \
        --nmesh "${TRANSFER_NMESH}" --lpt "${LPT}" --z "${Z}" \
        --method "${METHOD}" --seed "${SEED}" --nreal "${TRANSFER_NREAL}" \
        "${TRANSFER_PAIRED_FLAG[@]+"${TRANSFER_PAIRED_FLAG[@]}"}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" -o "${VLIB_CACHE_DIR}/transfer.npz"
    vlib::step_done transfer
fi

if vlib::step_check fairness "${VLIB_CACHE_DIR}/fairness.npz"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/fairness/run.py" \
        --Lcube "${LCUBE}" --aspect "${FAIR_ASPECT}" --nmesh "${FAIR_NMESH}" \
        --lpt "${LPT}" --z "${Z}" --seed "${SEED}" --nreal "${FAIR_NREAL}" \
        "${FAIR_PAIRED_FLAG[@]+"${FAIR_PAIRED_FLAG[@]}"}" \
        --taper-frac "${TAPER_FRAC}" --mu-split "${MU_SPLIT}" \
        --kmax-frac-ny "${KMAX_FRAC_NY}" -o "${VLIB_CACHE_DIR}/fairness.npz"
    vlib::step_done fairness
fi

if vlib::step_check plot_field "${SUPPLEMENTARY}/slab-anisotropy.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/field/plot.py" \
        -i "${VLIB_CACHE_DIR}/field.npz" -o "${SUPPLEMENTARY}/slab-anisotropy.pdf"
    vlib::step_done plot_field
fi

# shellcheck disable=SC2086
if vlib::step_check plot_transfer "${OUTPUT}/slab-transfer.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/transfer/plot.py" \
        -i "${VLIB_CACHE_DIR}/transfer.npz" --ylim ${YLIM} \
        --percent-band "${PERCENT_BAND}" --title "" -o "${OUTPUT}/slab-transfer.pdf"
    vlib::step_done plot_transfer
fi

if vlib::step_check plot_fairness "${OUTPUT}/slab-fairness.pdf"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/fairness/plot.py" \
        -i "${VLIB_CACHE_DIR}/fairness.npz" -o "${OUTPUT}/slab-fairness.pdf"
    vlib::step_done plot_fairness
fi

if vlib::step_check evaluate_transfer "${SUPPLEMENTARY}/transfer-result.json"; then
    VALIDATION_EVALUATION=report vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/transfer/evaluate.py" --archive "${VLIB_CACHE_DIR}/transfer.npz" \
        --figure "${OUTPUT}/slab-transfer.pdf" --output "${SUPPLEMENTARY}/transfer-result.json"
    vlib::step_done evaluate_transfer
fi

if vlib::step_check evaluate_fairness "${SUPPLEMENTARY}/fairness-result.json"; then
    VALIDATION_EVALUATION=report vlib::run_python "${STEPSIC_ENV}" \
        "${BASEDIR}/scripts/fairness/evaluate.py" --archive "${VLIB_CACHE_DIR}/fairness.npz" \
        --figure "${OUTPUT}/slab-fairness.pdf" --output "${SUPPLEMENTARY}/fairness-result.json"
    vlib::step_done evaluate_fairness
fi

if vlib::step_check evaluate "${OUTPUT}/result.json"; then
    vlib::run_python "${STEPSIC_ENV}" "${BASEDIR}/scripts/evaluate.py" \
        --result "${SUPPLEMENTARY}/transfer-result.json" \
        --result "${SUPPLEMENTARY}/fairness-result.json" \
        --data "${VLIB_CACHE_DIR}/transfer.npz" --data "${VLIB_CACHE_DIR}/fairness.npz" \
        --figure "${OUTPUT}/slab-transfer.pdf" --figure "${OUTPUT}/slab-fairness.pdf" \
        --figure "${SUPPLEMENTARY}/slab-anisotropy.pdf" --output "${OUTPUT}/result.json"
    vlib::step_done evaluate
fi

vlib::report_done
