#!/usr/bin/env bash
#*****************************************************************************#
#  run_profiles.sh - Profile stepsic with three complementary tools.
#
#  Prerequisites (install into the stepsic conda env):
#    pip install py-spy scalene snakeviz
#
#  Usage:
#    bash profiling/run-profiles.sh [small|medium|large|all|cprofile]
#
#  Note:
#    large is intentionally opt-in; all runs small + medium only.
#
#  Output:
#    profiling/audit/stepsic-small.svg          py-spy flamegraph (small)
#    profiling/audit/stepsic-medium.svg         py-spy flamegraph (medium)
#    profiling/audit/stepsic-large.svg          py-spy flamegraph (large)
#    profiling/audit/stepsic-scalene-small.json scalene line-level report (small)
#    profiling/audit/stepsic-scalene-medium.json scalene line-level report (medium)
#    profiling/audit/stepsic-scalene-large.json scalene line-level report (large)
#    profiling/audit/stepsic-small.prof         cProfile binary dump (small)
#    profiling/audit/stepsic-medium.prof        cProfile binary dump (medium)
#    profiling/audit/stepsic-large.prof         cProfile binary dump (large)
#    profiling/audit/cprofile-small.txt         cProfile text summary (small)
#    profiling/audit/cprofile-medium.txt        cProfile text summary (medium)
#    profiling/audit/cprofile-large.txt         cProfile text summary (large)
#    profiling/audit/output-small/               generated IC output (small)
#    profiling/audit/output-medium/              generated IC output (medium)
#    profiling/audit/output-large/               generated IC output (large)
#*****************************************************************************#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_DIR="${SCRIPT_DIR}/audit"
PROFILE_DIR="${AUDIT_DIR}"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# Resolve the stepsic entry point relative to the profile directory
# Adjust this if stepsic.py lives elsewhere
STEPSIC="${STEPSIC_ENTRY:-stepsic.py}"

SMALL_CFG="${CONFIG_DIR}/profile-small.toml"
MEDIUM_CFG="${CONFIG_DIR}/profile-medium.toml"
LARGE_CFG="${CONFIG_DIR}/profile-large.toml"

MODE="${1:-all}"

mkdir -p "${PROFILE_DIR}"

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "WARNING: '$1' not found. Install with: pip install $1"
        echo "         Skipping $1 profiling."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  1. cProfile - deterministic, zero overhead, cumulative call graph
# ---------------------------------------------------------------------------
run_cprofile() {
    local cfg="$1" label="$2"
    banner "cProfile - ${label}"

    python -m cProfile \
        -o "${PROFILE_DIR}/stepsic-${label}.prof" \
        "${STEPSIC}" "${cfg}"

    # Pretty-print the top 40 cumulative entries
    python - <<PY | tee "${PROFILE_DIR}/cprofile-${label}.txt"
import pstats
s = pstats.Stats("${PROFILE_DIR}/stepsic-${label}.prof")
s.strip_dirs().sort_stats("cumtime").print_stats(40)
print()
s.strip_dirs().sort_stats("tottime").print_stats(40)
PY

    echo ""
    echo "  Binary dump:  ${PROFILE_DIR}/stepsic-${label}.prof"
    echo "  Text summary: ${PROFILE_DIR}/cprofile-${label}.txt"
    echo "  Interactive:  snakeviz ${PROFILE_DIR}/stepsic-${label}.prof"
    echo ""
}

# ---------------------------------------------------------------------------
#  2. scalene - line-level CPU + memory + GPU profiling
#
#  scalene's CLI arg parser eats script arguments instead of forwarding
#  them, so we use thin wrapper scripts (scalene-*.py) that set sys.argv
#  explicitly and call runpy.run_path.
# ---------------------------------------------------------------------------
run_scalene() {
    local cfg="$1" label="$2"
    if ! check_tool "scalene"; then return; fi

    local wrapper="${SCRIPT_DIR}/scalene-wrapper.py"
    if [[ ! -f "${wrapper}" ]]; then
        echo "WARNING: scalene wrapper '${wrapper}' not found. Skipping."
        return
    fi

    banner "scalene - ${label}"

    local outfile="${PROFILE_DIR}/stepsic-scalene-${label}.json"
    STEPSIC_CONFIG="${cfg}" scalene run \
        --outfile "${outfile}" \
        --- "${wrapper}"

    echo ""
    echo "  Profile: ${outfile}"
    echo "  View:    scalene view ${outfile}"
    echo ""
}

# ---------------------------------------------------------------------------
#  3. py-spy - sampling profiler with flamegraph
#
#  py-spy needs ptrace access. Rather than running sudo py-spy (which
#  spawns the *profiled* process as root - breaking HOME, permissions,
#  and library caches), we relax ptrace_scope once and run everything
#  as the normal user.
# ---------------------------------------------------------------------------
run_pyspy() {
    local cfg="$1" label="$2"
    if ! check_tool "py-spy"; then return; fi
    banner "py-spy flamegraph - ${label}"

    local PYTHON_BIN
    PYTHON_BIN="$(command -v python)"

    local scope_file="/proc/sys/kernel/yama/ptrace_scope"
    local old_scope=""

    if [[ -r "${scope_file}" ]]; then
        old_scope="$(cat "${scope_file}")"

        if [[ "${old_scope}" -ne 0 ]]; then
            echo "  ptrace_scope is ${old_scope}; temporarily setting to 0."
            sudo sysctl kernel.yama.ptrace_scope=0 >/dev/null

            restore_ptrace_scope() {
                echo "  Restoring ptrace_scope=${old_scope}"
                sudo sysctl "kernel.yama.ptrace_scope=${old_scope}" >/dev/null || true
            }

            trap restore_ptrace_scope RETURN
        fi
    fi

    mkdir -p "${PROFILE_DIR}"

    # Try --native first; fall back to Python-only frames if it fails
    if ! py-spy record \
            --native \
            --rate 100 \
            --output "${PROFILE_DIR}/stepsic-${label}.svg" \
            -- "${PYTHON_BIN}" "${STEPSIC}" "${cfg}"; then

        echo "  --native failed; retrying without native stack unwinding..."
        py-spy record \
            --rate 100 \
            --output "${PROFILE_DIR}/stepsic-${label}.svg" \
            -- "${PYTHON_BIN}" "${STEPSIC}" "${cfg}"
    fi

    echo ""
    echo "  Flamegraph: ${PROFILE_DIR}/stepsic-${label}.svg"
    echo "  Open in browser for interactive exploration."
    echo ""
}


# ---------------------------------------------------------------------------
#  Run profiles
# ---------------------------------------------------------------------------
case "${MODE}" in
    small)
        run_cprofile  "${SMALL_CFG}"  "small"
        run_scalene   "${SMALL_CFG}"  "small"
        run_pyspy     "${SMALL_CFG}"  "small"
        ;;
    medium)
        run_cprofile  "${MEDIUM_CFG}" "medium"
        run_scalene   "${MEDIUM_CFG}" "medium"
        run_pyspy     "${MEDIUM_CFG}" "medium"
        ;;
    large)
        run_cprofile  "${LARGE_CFG}"  "large"
        run_scalene   "${LARGE_CFG}"  "large"
        run_pyspy     "${LARGE_CFG}"  "large"
        ;;
    all)
        # Small: all three tools
        run_cprofile  "${SMALL_CFG}"  "small"
        run_scalene   "${SMALL_CFG}"  "small"
        run_pyspy     "${SMALL_CFG}"  "small"
        # Medium: all three tools
        run_cprofile  "${MEDIUM_CFG}" "medium"
        run_scalene   "${MEDIUM_CFG}" "medium"
        run_pyspy     "${MEDIUM_CFG}" "medium"
        ;;
    cprofile)
        run_cprofile  "${SMALL_CFG}"  "small"
        run_cprofile  "${MEDIUM_CFG}" "medium"
        ;;
    *)
        echo "Usage: $0 [small|medium|large|all|cprofile]"
        exit 1
        ;;
esac

banner "Done"
echo "  Quick analysis commands:"
echo "    snakeviz profiling/audit/stepsic-small.prof"
echo "    snakeviz profiling/audit/stepsic-medium.prof"
echo "    snakeviz profiling/audit/stepsic-large.prof"
echo "    open profiling/audit/stepsic-medium.svg"
echo "    open profiling/audit/stepsic-large.svg"
echo "    scalene view profiling/audit/stepsic-scalene-small.json"
echo "    scalene view profiling/audit/stepsic-scalene-large.json"
echo ""
