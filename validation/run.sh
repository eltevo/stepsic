#!/usr/bin/env bash
# Run selected validation campaigns.
set -euo pipefail

VDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="${VDIR}/_common/profiles.py"
RESULTS="${VDIR}/_common/result.py"
SIZE="medium"
SIZE_EXPLICIT=0
CONFIG=""
EVALUATION="gate"
CAMPAIGN_SELECTION=""
GROUP=""
RUN_ALL=0
LIST_CAMPAIGNS=0
SIZE_COUNT=0
CONFIG_COUNT=0
EVALUATION_COUNT=0
CAMPAIGN_COUNT=0
GROUP_COUNT=0
ALL_COUNT=0

usage() {
    echo "Usage: bash validation/run.sh (--campaign=NAME,... | --group=paper|supplementary|extended | --all) [--size=small|medium|large] [--evaluation=skip|report|gate]"
    echo "       bash validation/run.sh --campaign=NAME --config=PROFILE.toml [--evaluation=skip|report|gate]"
    echo "       bash validation/run.sh --list-campaigns"
}

for arg in "$@"; do
    case "${arg}" in
        --size=*) SIZE="${arg#--size=}"; SIZE_EXPLICIT=1; (( SIZE_COUNT++ )) || true ;;
        --config=*) CONFIG="${arg#--config=}"; (( CONFIG_COUNT++ )) || true ;;
        --evaluation=*) EVALUATION="${arg#--evaluation=}"; (( EVALUATION_COUNT++ )) || true ;;
        --campaign=*) CAMPAIGN_SELECTION="${arg#--campaign=}"; (( CAMPAIGN_COUNT++ )) || true ;;
        --group=*) GROUP="${arg#--group=}"; (( GROUP_COUNT++ )) || true ;;
        --all) RUN_ALL=1; (( ALL_COUNT++ )) || true ;;
        --list-campaigns) LIST_CAMPAIGNS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: ${arg}" >&2; usage >&2; exit 2 ;;
    esac
done

if (( SIZE_COUNT > 1 || CONFIG_COUNT > 1 || EVALUATION_COUNT > 1 \
      || CAMPAIGN_COUNT > 1 || GROUP_COUNT > 1 || ALL_COUNT > 1 )); then
    echo "ERROR: selection and configuration options may be specified only once." >&2
    exit 2
fi

case "${EVALUATION}" in skip|report|gate) ;; *) echo "ERROR: invalid evaluation mode: ${EVALUATION}" >&2; exit 2 ;; esac

CATALOG="$(conda run --no-capture-output -n stepsic python "${RESOLVER}" catalog)"
if (( LIST_CAMPAIGNS )); then
    printf '%s\n' "${CATALOG}"
    exit 0
fi

selection_count=0
[[ -n "${CAMPAIGN_SELECTION}" ]] && (( selection_count++ )) || true
[[ -n "${GROUP}" ]] && (( selection_count++ )) || true
(( RUN_ALL )) && (( selection_count++ )) || true
if (( selection_count != 1 )); then
    echo "ERROR: select exactly one of --campaign, --group, or --all." >&2
    usage >&2
    exit 2
fi
case "${GROUP:-paper}" in paper|supplementary|extended) ;; *) echo "ERROR: unknown group: ${GROUP}" >&2; exit 2 ;; esac
if [[ -n "${CONFIG}" && ( ${SIZE_EXPLICIT} -eq 1 || -n "${GROUP}" || ${RUN_ALL} -eq 1 ) ]]; then
    echo "ERROR: --config is mutually exclusive with --size and requires one --campaign." >&2
    exit 2
fi

declare -A ENTRYPOINTS=()
declare -A ROLES=()
declare -A EVALUATES=()
declare -a ALL=()
while IFS=$'\t' read -r name entrypoint role prerequisites evaluates; do
    [[ -z "${name}" ]] && continue
    ENTRYPOINTS["${name}"]="${entrypoint}"
    ROLES["${name}"]="${role}"
    EVALUATES["${name}"]="${evaluates}"
    ALL+=("${name}")
done <<< "${CATALOG}"

declare -a SELECTED=()
if (( RUN_ALL )); then
    SELECTED=("${ALL[@]}")
elif [[ -n "${GROUP}" ]]; then
    for name in "${ALL[@]}"; do
        [[ "${ROLES[${name}]}" == "${GROUP}" ]] && SELECTED+=("${name}")
    done
else
    IFS=',' read -r -a SELECTED <<< "${CAMPAIGN_SELECTION}"
fi

declare -A SEEN=()
for name in "${SELECTED[@]}"; do
    if [[ -z "${name}" || -z "${ENTRYPOINTS[${name}]+x}" ]]; then
        echo "ERROR: unknown campaign: ${name:-<empty>}" >&2
        exit 2
    fi
    if [[ -n "${SEEN[${name}]+x}" ]]; then
        echo "ERROR: duplicate campaign: ${name}" >&2
        exit 2
    fi
    SEEN["${name}"]=1
done
if [[ -n "${CONFIG}" && ${#SELECTED[@]} -ne 1 ]]; then
    echo "ERROR: a custom profile can select exactly one campaign." >&2
    exit 2
fi
[[ -n "${CONFIG}" ]] && SIZE="custom"

for name in "${SELECTED[@]}"; do
    if [[ -n "${CONFIG}" ]]; then
        conda run --no-capture-output -n stepsic python "${RESOLVER}" resolve \
            --config "${CONFIG}" --campaign "${name}" > /dev/null
    else
        conda run --no-capture-output -n stepsic python "${RESOLVER}" resolve \
            --size "${SIZE}" --campaign "${name}" > /dev/null
    fi
done

total="${#SELECTED[@]}"
index=0
summary_args=()
overall_status=0
for name in "${SELECTED[@]}"; do
    (( index++ )) || true
    echo ""
    echo "[${index}/${total}] ${name} (${SIZE})"
    args=(--evaluation="${EVALUATION}")
    if [[ -n "${CONFIG}" ]]; then args+=(--config="${CONFIG}"); else args+=(--size="${SIZE}"); fi
    run_status=0
    bash "${VDIR}/${ENTRYPOINTS[${name}]}" "${args[@]}" || run_status=$?
    (( run_status != 0 )) && overall_status=1
    if [[ "${EVALUATION}" != skip && "${EVALUATES[${name}]}" == true ]]; then
        campaign_dir="${ENTRYPOINTS[${name}]%/*}"
        if [[ -n "${CONFIG}" ]]; then
            profile_sha256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
            result_path="${VDIR}/${campaign_dir}/runs/custom/${profile_sha256}/output/result.json"
        else
            result_path="${VDIR}/${campaign_dir}/runs/${SIZE}/output/result.json"
        fi
        summary_args+=(--entry "${name}" "${run_status}" "${result_path}")
    fi
done

if (( ${#summary_args[@]} > 0 )); then
    python3 "${RESULTS}" summary --size "${SIZE}" --evaluation "${EVALUATION}" \
        "${summary_args[@]}" || overall_status=1
fi
exit "${overall_status}"
