# ============================================================================
# Cosmology helpers
# ============================================================================

_VLIB_COSMOLOGY_NAME="Planck2018EE+BAO"
_VLIB_COSMOLOGY_FILE="${_VLIB_COMMON_DIR}/cosmology/${_VLIB_COSMOLOGY_NAME}.toml"
declare -ag _VLIB_COSMOLOGY_KEYS=(
    H0 OMEGA_M OMEGA_B OMEGA_L NS AS SIGMA8 YHE MNU NNU NUR ZREI TCMB
    KPIVOT W0 WA
)

# Load the shared validation cosmology without replacing explicit overrides.
vlib::cosmology::load_defaults() {
    local requested="${COSMOLOGY_NAME:-${_VLIB_COSMOLOGY_NAME}}"
    if [[ "${requested}" != "${_VLIB_COSMOLOGY_NAME}" ]]; then
        echo "ERROR: unsupported validation cosmology: ${requested}" >&2
        return 2
    fi
    if [[ ! -f "${_VLIB_COSMOLOGY_FILE}" ]]; then
        echo "ERROR: validation cosmology file not found: ${_VLIB_COSMOLOGY_FILE}" >&2
        return 1
    fi

    local -A values=()
    local line key value required allowed
    while IFS= read -r line; do
        line="${line%%#*}"
        [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
        if [[ ! "${line}" =~ ^[[:space:]]*([A-Z][A-Z0-9_]*)[[:space:]]*=[[:space:]]*([-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?)[[:space:]]*$ ]]; then
            echo "ERROR: invalid validation cosmology line: ${line}" >&2
            return 2
        fi
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        allowed=0
        for required in "${_VLIB_COSMOLOGY_KEYS[@]}"; do
            [[ "${key}" == "${required}" ]] && allowed=1
        done
        if (( ! allowed )); then
            echo "ERROR: unknown validation cosmology key: ${key}" >&2
            return 2
        fi
        if [[ -n "${values[${key}]+set}" ]]; then
            echo "ERROR: duplicate validation cosmology key: ${key}" >&2
            return 2
        fi
        values["${key}"]="${value}"
    done < "${_VLIB_COSMOLOGY_FILE}"

    local variable
    for key in "${_VLIB_COSMOLOGY_KEYS[@]}"; do
        if [[ -z "${values[${key}]+set}" ]]; then
            echo "ERROR: missing validation cosmology key: ${key}" >&2
            return 2
        fi
        variable="COSMO_${key}"
        if [[ -z "${!variable:-}" ]]; then
            printf -v "${variable}" '%s' "${values[${key}]}"
        fi
        export "${variable}"
    done
    COSMOLOGY_NAME="${_VLIB_COSMOLOGY_NAME}"
    export COSMOLOGY_NAME
}

# EdS H0 for the glass-making StePS run.
#
# Always returns 100 km/s/Mpc (h_EdS = 1), independently of the target
# LCDM cosmology. Rationale: stepsic ICs use an h-independent RHO_CRIT
# (canonical H0 = 100 km/s/Mpc), so for StePS's rho_part / rho_cosm
# check to balance, the glass-run HubbleConstant must also use h = 1.
# The physically-motivated choice H0_EdS = H0_LCDM * sqrt(Omega_m_LCDM)
# (which holds the absolute matter density fixed in physical units) leaves
# h_EdS != 1, which exposes a unit mismatch in StePS's PERIODIC_Z mass
# check (Rsim is not converted by H_INDEPENDENT_UNITS=1 while L is) and
# trips the consistency assertion with rho_part / rho_cosm = 1 / h_EdS^2.
# H0 affects only the simulation timescale; comoving relaxation is
# invariant, so picking h = 1 is safe for glass-making.
#
# Usage: vlib::cosmology::eds_h0 <H0> <Omega_m>   (args ignored)
vlib::cosmology::eds_h0() {
    echo "100.0"
}
