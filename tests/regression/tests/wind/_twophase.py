"""Shared helpers of the two-phase wind tests (wind-design.md Sec. 10)."""

import re

# Defaults of src/wind_physics.h and src/wind.c
WARM_SLOPE = -0.44
WARM_NORM = {"K20": -0.961, "TIGRESSpp": -0.628}
HOT_M_NORM = {"K20": -0.86, "TIGRESSpp": -0.86 + -0.28819}  # log10(0.515)
HOT_E_NORM = {"K20": -0.70, "TIGRESSpp": -0.70 + -0.39794}  # log10(0.40)
HOT_M_SLOPE, HOT_E_SLOPE = -0.07, 0.14
CR_ETA_E = {"K20": 0.0, "TIGRESSpp": 0.3}

PHASE_KEYS = ["t", "n_hot", "m_hot", "n_warm", "m_warm", "E_cr_launch",
              "E_cr_release", "E_cr_deposit", "n_apex", "m_over_eta_hot",
              "m_over_eta_warm"]


def powerlaw(norm, slope, sigma_sfr):
    s = min(max(sigma_sfr, 1e-4), 1.0)
    return 10.0 ** norm * s ** slope


def phases_tail(log_path):
    """The last 'Wind phases' line of a SWIFT log as a dict, or None."""
    last = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Wind phases at" in line:
                last = line
    if last is None:
        return None
    nums = re.findall(r"=\s*([-+0-9.eE]+)", last)
    return dict(zip(PHASE_KEYS, map(float, nums)))
