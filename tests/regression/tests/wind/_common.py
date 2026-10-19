"""Shared set-up for the decoupled-wind tests (src/wind/TIGRESS/).

Build: PRFM star formation and entropy floor with the TIGRESS wind
(--with-wind=TIGRESS needs PRFM star formation and SPHENIX, the default
hydro scheme). Particles can be decoupled by hand through the optional
initial-conditions field WindDecoupled (int8): such a particle is launched
at the start time with its initial velocity and internal energy.
"""

from swiftreg import params, units

WIND_BUILD = [params.PRFM_COOLING_FLAG, "--with-chemistry=EAGLE",
              "--with-tracers=EAGLE", "--with-stars=EAGLE",
              "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
              "--with-ext-potential=constant", "--with-wind=TIGRESS"]

# Hydro and a constant external field; no cooling (--temperature is needed by
# the EAGLE tracers and computes temperatures without cooling)
NOSF_FLAGS = ["--hydro", "--external-gravity", "--temperature"]

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


WIND_TOTALS_KEYS = [
    "t", "n_launch", "m_launch", "E_kin_launch", "E_th_launch",
    "n_recouple", "m_recouple", "E_kin_recouple", "E_th_recouple",
    "E_pot_recouple", "E_launch_recouple", "E_launch_flight",
    "m_tracer_star", "E_tracer_star", "E_pot_tracer_star", "E_launch_tracer_star"]


def wind_totals(log_path):
    """Every 'Wind totals' line of a SWIFT log, as a list of dicts."""
    import re
    rows = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Wind totals at" in line:
                nums = re.findall(r"=\s*([-+0-9.eE]+)", line)
                rows.append(dict(zip(WIND_TOTALS_KEYS, map(float, nums))))
    return rows


def wind_tail(log_path):
    """The last 'Wind totals' line of a SWIFT log as a dict, or None."""
    rows = wind_totals(log_path)
    return rows[-1] if rows else None
