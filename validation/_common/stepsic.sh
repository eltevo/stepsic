# ============================================================================
# Stepsic TOML generator
# ============================================================================
#
# Write a stepsic configuration TOML file.
# Usage: vlib::stepsic::write_toml <output_path> [KEY=VALUE ...]
#
# Cosmology name comes from COSMOLOGY_NAME (set by vlib::cosmology::load_defaults).
# Geometry parameters (R_3D, D_4D, RCRIT, NRBINS, NSHELL, etc.) come from
# the calling environment; override any by passing KEY=VALUE pairs.
#
# Example:
#   vlib::stepsic::write_toml "${TOML_DIR}/ic.toml" \
#       GEOMETRY=spherical LBOX=500 R_3D=500 IC_DIR="${CACHE}/ic"
vlib::stepsic::write_toml() {
    local path="${1}"
    shift

    # Collect explicit key=value overrides into an associative array.
    declare -A _t
    local pair
    for pair in "$@"; do
        _t["${pair%%=*}"]="${pair#*=}"
    done

    # Resolve a key: override wins, then env-var fallback, then hard default.
    local _tv
    _tv() {
        local k="${1}" d="${2:-}"
        if [[ -n "${_t[${k}]+x}" ]]; then
            echo "${_t[${k}]}"
        else
            local ev="${!k:-}"
            echo "${ev:-${d}}"
        fi
    }

    local GEOMETRY; GEOMETRY="$(_tv GEOMETRY spherical)"
    local LBOX;     LBOX="$(_tv LBOX 500)"
    local PERIODIC; PERIODIC="$(_tv PERIODIC false)"
    local R_3D_V;   R_3D_V="$(_tv R_3D "${R_3D:-500}")"
    local D_4D_V;   D_4D_V="$(_tv D_4D "${D_4D:-75}")"
    local RCRIT_V;  RCRIT_V="$(_tv RCRIT "${RCRIT:-0}")"
    local BIN_MODE; BIN_MODE="$(_tv BIN_MODE omega)"
    local NRBINS_V; NRBINS_V="$(_tv NRBINS "${NRBINS:-224}")"
    local IC_TYPE;  IC_TYPE="$(_tv TYPE preglass)"
    local NGRID_V;  NGRID_V="$(_tv NGRID "${NGRID:-32}")"
    local NPART_V;  NPART_V="$(_tv NPART "${NPART:-0}")"
    local NSHELL_V; NSHELL_V="$(_tv NSHELL "${NSHELL:-12288}")"
    local IC_DIR_V; IC_DIR_V="$(_tv IC_DIR "${IC_DIR:-output}")"
    local IC_PFX;   IC_PFX="$(_tv IC_PREFIX stepsic)"
    local INPUT_GLASS_V; INPUT_GLASS_V="$(_tv INPUT_GLASS none)"
    local SPECTRUM; SPECTRUM="$(_tv SPECTRUM camb)"
    local NONLINEAR;NONLINEAR="$(_tv NONLINEAR false)"
    local HALOFIT;  HALOFIT="$(_tv HALOFIT mead2020)"
    local SEED_V;   SEED_V="$(_tv SEED 42)"
    local LPTORDER; LPTORDER="$(_tv LPTORDER 0)"
    local NMESH_V;  NMESH_V="$(_tv NMESH 32)"
    local REDSHIFT; REDSHIFT="$(_tv REDSHIFT 0)"
    local INTERP;   INTERP="$(_tv INTERPOLATION cic)"
    local COMPENSATE; COMPENSATE="$(_tv COMPENSATE false)"
    local SPHEREMODE; SPHEREMODE="$(_tv SPHEREMODE false)"
    local PAIRED_V; PAIRED_V="$(_tv PAIRED false)"
    local FIXED_V;  FIXED_V="$(_tv FIXED false)"
    local NMS_V;    NMS_V="$(_tv NMESHSAMPLES 1)"
    local PHASE_SHIFT_V; PHASE_SHIFT_V="$(_tv PHASE_SHIFT 0.0)"
    local ROTATE_V; ROTATE_V="$(_tv ROTATE 0.0)"
    local USE_DOUBLE_V; USE_DOUBLE_V="$(_tv USE_DOUBLE false)"
    local SAVE_WHITE_NOISE_V; SAVE_WHITE_NOISE_V="$(_tv SAVE_WHITE_NOISE false)"
    # Optional cosmological overrides (empty = use cosmology.toml defaults).
    # Set OMEGA_M=1.0 and OMEGA_L=0.0 for EdS pre-glass ICs so that particle
    # masses satisfy StePS's h-independent mass-consistency check (rho_part /
    # rho_cosm = 1). When empty, the COSMOLOGY block in cosmology.toml supplies
    # the values via setdefault and these lines are omitted from the TOML.
    local OMEGA_M_V; OMEGA_M_V="$(_tv OMEGA_M "")"
    local OMEGA_L_V; OMEGA_L_V="$(_tv OMEGA_L "")"
    local OMEGA_B_V; OMEGA_B_V="$(_tv OMEGA_B "")"
    local H0_V; H0_V="$(_tv H0 "")"

    mkdir -p "$(dirname "${path}")"
    vlib::atomic_text "${path}" <<EOF2
GEOMETRY = "${GEOMETRY}"
LBOX = ${LBOX}
PERIODIC = ${PERIODIC}
R_3D = ${R_3D_V}
D_4D = ${D_4D_V}
RCRIT = ${RCRIT_V}
BIN_MODE = "${BIN_MODE}"
NRBINS = ${NRBINS_V}
COI = [0, 0, 0]
TYPE = "${IC_TYPE}"
NGRID = ${NGRID_V}
NPART = ${NPART_V}
NSHELL = ${NSHELL_V}
INPUT_GLASS = "${INPUT_GLASS_V}"
IC_DIR = "${IC_DIR_V}"
IC_PREFIX = "${IC_PFX}"
COSMOLOGY = "${COSMOLOGY_NAME}"
${OMEGA_M_V:+OMEGA_M = ${OMEGA_M_V}}
${OMEGA_L_V:+OMEGA_L = ${OMEGA_L_V}}
${OMEGA_B_V:+OMEGA_B = ${OMEGA_B_V}}
${H0_V:+H0 = ${H0_V}}
SPECTRUM = "${SPECTRUM}"
NONLINEAR = ${NONLINEAR}
HALOFIT = "${HALOFIT}"
SEED = ${SEED_V}
COMOVING = true
HINDEPENDENT = true
LPTORDER = ${LPTORDER}
NMESH = ${NMESH_V}
REDSHIFT = ${REDSHIFT}
INTERPOLATION = "${INTERP}"
COMPENSATE = ${COMPENSATE}
SPHEREMODE = ${SPHEREMODE}
PAIRED = ${PAIRED_V}
FIXED = ${FIXED_V}
PHASE_SHIFT = ${PHASE_SHIFT_V}
NMESHSAMPLES = ${NMS_V}
ROTATE = ${ROTATE_V}
IC_FORMAT = "hdf5"
USE_DOUBLE = ${USE_DOUBLE_V}
SAVE_WHITE_NOISE = ${SAVE_WHITE_NOISE_V}
UNIT_L_IN_CM = 3.085678e24
UNIT_M_IN_G = 1.989e44
UNIT_V_IN_KMPS = 20.738652969844207
INPUT_SPECTRUM = "none"
INPUT_SPECTRUM_UNIT_L_IN_CM = 3.085678e24
EOF2

    unset -f _tv
}
