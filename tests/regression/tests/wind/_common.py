"""Shared set-up for the decoupled-wind tests (src/wind/TIGRESS/).

Build: PRFM star formation and entropy floor with the TIGRESS wind
(--with-wind=TIGRESS needs PRFM star formation and SPHENIX, the default
hydro scheme). Particles can be decoupled by hand through the optional
initial-conditions field WindDecoupled (int8): such a particle is launched
at the start time with its initial velocity and internal energy. Star
particles flagged WindCarrier (int8) in the ICs are wind carriers launched at
the start time, with internal energy WindCarrierInternalEnergies.
"""

import numpy as np

from swiftreg import params, units

WIND_BUILD = [params.PRFM_COOLING_FLAG, "--with-chemistry=EAGLE",
              "--with-tracers=EAGLE", "--with-stars=EAGLE",
              "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
              "--with-ext-potential=constant", "--with-wind=TIGRESS"]

# Hydro and a constant external field; no cooling (--temperature is needed by
# the EAGLE tracers and computes temperatures without cooling)
NOSF_FLAGS = ["--hydro", "--external-gravity", "--temperature"]

# Wind carriers (Wind:mode = carrier) are star particles that deposit through
# the stars density and feedback loops, which exist only with --feedback
# (feedback model none: ordinary stars do nothing there)
CARRIER_FLAGS = NOSF_FLAGS + ["--stars", "--feedback", "--limiter", "--sync"]

KERNEL_GAMMA = 1.825742  # cubic spline in 3D (default kernel)


def carrier_params(d_rec_kpc, t_rec_myr, n_c=1.0, model="none", **extra):
    """Wind section for carriers; loading model none launches nothing (the
    tests that place carriers in the ICs)."""
    w = {"loading_model": model, "mode": "carrier", "carriers_per_particle": n_c,
         "recoupling_distance_kpc": d_rec_kpc, "recoupling_time_Myr": t_rec_myr}
    w.update(extra)
    return w


def carrier_events(path):
    """Rows of wind_carriers.txt (one per deposit) as dicts."""
    out = []
    try:
        rows = np.atleast_2d(np.loadtxt(path, comments="#"))
    except (OSError, ValueError):
        return out
    if rows.size == 0:
        return out
    for r in rows:
        out.append({"id": int(r[0]), "t_launch": r[1], "t_deposit": r[2],
                    "criterion": int(r[3]), "x_launch": r[4:7].tolist(),
                    "x_deposit": r[7:10].tolist(), "distance": r[10],
                    "d_rec": r[11], "mass": r[12], "v": r[13:16].tolist(),
                    "u": r[16], "h": r[17], "n_ngb": int(r[18]),
                    "weight_sum": r[19]})
    return out


def carrier_tail(log_path):
    """The last 'Wind carrier totals' line of a SWIFT log as a dict, or None."""
    import re
    last = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Wind carrier totals at" in line:
                last = line
    if last is None:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?",
                                         last.split("Wind carrier totals at t=")[1])]
    keys = ["t", "n_launch", "m_launch", "px_launch", "py_launch", "pz_launch",
            "E_kin_launch", "E_th_launch", "E_net_launch", "MZ_launch", "n_failed",
            "n_deposit", "n_by_time", "m_deposit", "px_deposit", "py_deposit",
            "pz_deposit", "E_kin_deposit", "E_th_deposit", "E_pot_deposit",
            "E_floor", "MZ_deposit", "n_retry"]
    return dict(zip(keys, nums))

MYR = units.MYR / units.UNIT_TIME
KPC = 1.0
KMS = 1.0
# Internal acceleration unit (km/s)^2 / kpc in cm/s^2
ACC_UNIT = units.UNIT_VELOCITY ** 2 / units.UNIT_LENGTH


def base_params(ctx, ic_file, t_end, snapshot_dt, dt_max, g_cgs=(0.0, 0.0, 0.0)):
    """eagle_base without the EAGLE subgrid sections, with the PRFM floor
    (threshold 0.13 cm^-3, inactive in the low-density boxes used here)."""
    p = params.eagle_base(ic_file, t_end, ctx.data_dir, snapshot_dt, dt_max=dt_max,
                          cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["ConstantPotential"] = {"g_cgs": list(g_cgs)}
    p["PRFMEntropyFloor"] = {"calibration": "OK22",
                             "density_threshold_H_p_cm3": 0.13}
    return p


def wind_tail(log_path):
    """The last 'Wind totals' line of a SWIFT log as a dict, or None."""
    last = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Wind totals at" in line:
                last = line
    if last is None:
        return None
    import re
    nums = re.findall(r"=\s*([-+0-9.eE]+)", last)
    keys = ["t", "n_launch", "m_launch", "E_kin_launch", "E_th_launch",
            "n_recouple", "m_recouple", "E_kin_recouple", "E_th_recouple"]
    return dict(zip(keys, map(float, nums)))
