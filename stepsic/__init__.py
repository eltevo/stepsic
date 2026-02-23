'''
StePS initial-condition generator package.
'''

from textwrap import dedent
import subprocess

def get_git_revision_hash() -> str:
    try:
        # Returns the full hash (e.g., 'a1b2c3d4...')
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"
def get_git_short_hash() -> str:
    try:
        # Returns the short 7-character hash
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"
def get_git_branch() -> str:
    try:
        # Returns the branch name (e.g., 'main', 'develop', or 'feature/physics-fix')
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

__version__ = '2.0.0'
__year__ = '2017-2026'
__authors__ = ['Balazs Pal', 'Gabor Racz']
__programname__ = "stepsic.py"
__githash__ = get_git_revision_hash()
__gitshorthash__ = get_git_short_hash()
__gitbranch__ = get_git_branch()


def _make_header(Nart: int = 79, Ncop: int = 79, Nwar: int = 79):
    art = dedent(fr'''
    \t     _                 _      
    \t    | |               (_)     
    \t ___| |_ ___ _ __  ___ _  ___ 
    \t/ __| __/ _ \ '_ \/ __| |/ __|
    \t\__ \ ||  __/ |_) \__ \ | (__ 
    \t|___/\__\___| .__/|___/_|\___|
    \t            | |               
    \t            |_|               
    {__programname__} {__version__} (branch: {__gitbranch__}; git rev.: {__gitshorthash__})
    \tAn IC generator python script for
    \tSTEreographically Projected cosmological Simulations
    ''')
    # cop = dedent(f'''
    # Copyright (C) ({__year__}) {', '.join(__authors__)}
    # \tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA
    # \tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
    # \tHeavy-ion Physics Research Group, HUN-REN Wigner RCP | Budapest, Hungary
    # \tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
    # \tDepartment of Physics, University of Helsinki | Helsinki, Finland
    # ''')
    cop = dedent(f'''
    Copyright (C) ({__year__}) {', '.join(__authors__)}
    \tDepartment of Physics of Complex Systems, Eotvos Lorand University
    \tHeavy-ion Physics Research Group, HUN-REN Wigner RCP
    \tDepartment of Physics, University of Helsinki
    ''')
    war = dedent(f'''
    stepsic comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions. See the LICENSE file for details.
    ''')
    # Define horizontal borders: +-- ... --+
    b  = lambda N: f'+{"-"*(N)}+'
    # Converts multiline string to list of lines
    untab = lambda s: s.replace(r'\t', '\t')   # Replace literal `\t` with tabs
    ls = lambda s: untab(s).expandtabs(4).splitlines()[1:]
    # Pad RHS of all lines with spaces to get them equally `N` chars wide
    T  = lambda s, N: '\n'.join([f"| {l}{' '*(N-1-len(l))}|" for l in ls(s)])

    return ('\n'.join([
        f'{b(Nart)}\n{T(art, Nart)}\n{b(Nart)}',
        f'{b(Ncop)}\n{T(cop, Ncop)}\n{b(Ncop)}',
        f'{b(Nwar)}\n{T(war, Nwar)}\n{b(Nwar)}'
    ]))

__header__ = _make_header()