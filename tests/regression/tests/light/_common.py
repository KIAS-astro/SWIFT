"""Shared set-up for the light wind particle tests (Wind:mode light).

Builds: the wind build of tests/wind (PRFM, EAGLE subgrid modules, TIGRESS
wind) with SPHENIX, and the same with GIZMO-MFM. Light particles are
spawned at tree rebuilds from per-particle reservoirs and merged back by the
refinement clock of Bennett et al. (2025); design in
light-particles-design.md Sec. 8.
"""

import re

from ..wind import _common as wind_common

SPH_BUILD = list(wind_common.WIND_BUILD)
MFM_BUILD = list(wind_common.WIND_BUILD) + ["--with-hydro=gizmo-mfm",
                                            "--with-riemann-solver=hllc"]

LIGHT_KEYS = ["t", "m_added", "E_added", "m_to_stars", "m_rounding", "n_spawn", "m_spawn",
              "m_source", "n_merge", "m_merge", "E_heat", "m_held", "n_light",
              "m_light", "n_light_decoupled", "Px_launch", "Py_launch",
              "Pz_launch"]


def light_lines(log_path):
    """All 'Light totals' lines of a SWIFT log as a list of dicts."""
    out = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Light totals at" in line:
                nums = re.findall(r"=\s*([-+0-9.eE]+)", line)
                out.append(dict(zip(LIGHT_KEYS, map(float, nums))))
    return out


def light_tail(log_path):
    lines = light_lines(log_path)
    return lines[-1] if lines else None


def light_params(m_g_Msun, N, **extra):
    """The Wind section entries of light mode."""
    p = {"mode": "light", "light_gas_mass_Msun": float(m_g_Msun),
         "light_N": float(N)}
    p.update(extra)
    return p
