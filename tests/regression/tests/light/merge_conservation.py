"""Merging of light wind particles conserves mass, momentum and energy.

A periodic box of 2 kpc holds 16^3 gas particles of m_g on a lattice
(n_H = 0.01 cm^-3, T = 1e4 K, at rest) and 800 coupled light particles of
m_l = m_g / 10 at random positions, flagged WindLight with given dyes, moving
at random velocities of 20 km/s rms per component with T = 1e5 K. No
gravity, cooling or star formation. Half the light particles start with
f_w = 0.05 (clock run out: target m_g) and half with f_w = 1 (target
m_g (0.1 / f_w)^1/2, rising to m_g after t_0 ln 10 = 23 Myr). Tree rebuilds
are forced every 0.5 Myr, and merging happens at each.

Two runs per hydro scheme (SPHENIX; GIZMO-MFM with --limiter --sync):

  short: 4 steps of 1e-4 Myr; the first merges happen at the rebuild of
    step 2. Between the initial conditions and the end, with N_merge > 0:
      total mass to 1e-6 (float masses);
      momentum change over sum m |v| to 1e-5;
      total energy (kinetic + internal) change over the total to 1e-5;
    the hydrodynamical change over 4e-4 Myr is below 1e-7 of each total,
    so the tolerances measure the float storage of the merged state.
  long: 50 Myr. Total mass to 1e-6; momentum change over sum m |v| and
    energy change over the total each no larger than 2 times that of a
    control run on the same initial conditions with Wind:mode particle (no
    merging), plus 1e-5. SPHENIX with individual time-steps conserves
    momentum only to about 1e-3 here (control). For momentum the bound is at
    least 5e-4: snapshot velocities of particles in mid-step are
    extrapolated with their accelerations, and in the MFM run with merging
    sum m v read from snapshots every 0.5 Myr moves between 7e-6 and 3e-4 of
    sum m |v| without a trend (measured 2026-10-20), while the short run
    shows each merge conserving it to 1e-8. Every clock has run out by 50 Myr (f_w = 1 falls to 0.1 at
    23 Myr), so no particle below m_g / 2 may still be flagged light, and
    the largest mass is recorded against 3/2 m_g (design Sec. 4.2).
"""

import os

import numpy as np

from swiftreg import ics, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..wind import _common as wind_common
from . import _common

DESCRIPTION = "light merging: mass, momentum, total energy conserved; clocks run out"
TIER = "quick"
BUILDS = {"sph": _common.SPH_BUILD, "mfm": _common.MFM_BUILD}

N_SIDE = 16
BOX = 2.0  # kpc
NH = 0.01
N_LIGHT_PARTS = 800
N_RATIO = 10
SIGMA_V = 20.0  # km/s
T_HEAVY, T_LIGHT = 1e4, 1e5
MYR = units.MYR / units.UNIT_TIME


def _write_ics(path):
    gas, m_g_Msun = ics.uniform_gas(N_SIDE, BOX, NH, T_HEAVY, mu=units.MU_IONISED)
    rng = np.random.default_rng(7)
    n0 = len(gas["Masses"])
    nl = N_LIGHT_PARTS
    m_l = gas["Masses"][0] / N_RATIO
    u_l = units.u_from_T(T_LIGHT, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    light = {
        "Coordinates": rng.uniform(0, BOX, (nl, 3)),
        "Velocities": rng.normal(0, SIGMA_V, (nl, 3)),
        "Masses": np.full(nl, m_l),
        "SmoothingLength": np.full(nl, gas["SmoothingLength"][0]),
        "InternalEnergy": np.full(nl, u_l),
        "ParticleIDs": np.arange(n0 + 1, n0 + nl + 1, dtype=np.uint64),
    }
    out = {k: np.concatenate([gas[k], light[k]]) for k in gas}
    out["WindLight"] = np.concatenate([np.zeros(n0, np.int8), np.ones(nl, np.int8)])
    dye = np.where(np.arange(nl) % 2 == 0, 0.05, 1.0).astype(np.float32)
    out["WindDye"] = np.concatenate([np.zeros(n0, np.float32), dye])
    ics.write(path, BOX, gas=out)
    return m_g_Msun


def _params(ctx, t_end, dt_max, mode):
    p = wind_common.base_params(ctx, "ics.hdf5", t_end, t_end, dt_max)
    # Three top-level cells of 0.67 kpc: kernel_gamma h_max must stay below
    # the cell size while merged particles' h grows
    p["Scheduler"]["max_top_level_cells"] = 3
    p["SPH"]["h_max"] = 0.3
    p["Wind"] = {"loading_model": "none",
                 **_common.light_params(ctx.info["m_g_Msun"], N_RATIO,
                                        light_rebuild_interval_Myr=0.5)}
    if mode == "particle":
        p["Wind"] = {"loading_model": "none", "mode": "particle"}
    return p


def run(ctx):
    ctx.table(wind_common.params.PRFM_COOLING_TABLE)
    ctx.info["m_g_Msun"] = _write_ics(ctx.path("ics.hdf5"))
    for b in BUILDS:
        flags = list(wind_common.NOSF_FLAGS)
        if b == "mfm":
            flags += ["--limiter", "--sync"]
        for name, t_end, dt_max, mode in [("short", 4e-4 * MYR, 1e-4 * MYR, "light"),
                                          ("long", 50 * MYR, 0.5 * MYR, "light"),
                                          ("control", 50 * MYR, 0.5 * MYR, "particle")]:
            sub = f"{b}_{name}"
            os.makedirs(ctx.path(sub), exist_ok=True)
            if not os.path.exists(ctx.path(sub, "ics.hdf5")):
                os.symlink(ctx.path("ics.hdf5"), ctx.path(sub, "ics.hdf5"))
            ctx.swift(_params(ctx, t_end, dt_max, mode), flags, binary=b, subdir=sub,
                      name="run", timeout=1800)


def _totals(snap):
    m = snap.read(0, "Masses", cgs=False)
    v = snap.read(0, "Velocities", cgs=False)
    u = snap.read(0, "InternalEnergies", cgs=False)
    P = (m[:, None] * v).sum(axis=0)
    E = 0.5 * (m * (v ** 2).sum(axis=1)).sum() + (m * u).sum()
    return {"M": m.sum(), "P": P, "Pscale": (m * np.linalg.norm(v, axis=1)).sum(),
            "E": E, "m": m}


def analyze(ctx):
    checks = []
    m_g = ctx.info["m_g_Msun"] * units.SOLAR_MASS / units.UNIT_MASS
    for b in BUILDS:
        res = {}
        for name in ("short", "long", "control"):
            snaps = snapshots(ctx.path(f"{b}_{name}"))
            res[name] = (_totals(snaps[0]), _totals(snaps[-1]), snaps[-1])
        a, z, last = res["short"]
        tail = _common.light_tail(ctx.path(f"{b}_short", "run.log"))
        n_merge = tail["n_merge"] if tail else 0
        checks.append(Check(f"{b} short: merges", n_merge, 1, kind="min",
                            source="light particles below half their target"))
        checks.append(Check(f"{b} short: mass", z["M"], a["M"], kind="rel", tol=1e-6,
                            source="merge adds masses; float storage"))
        checks.append(Check(f"{b} short: |dP| / sum m|v|",
                            np.abs(z["P"] - a["P"]).max() / a["Pscale"], 1e-5, kind="max",
                            source="merge conserves m v exactly; float storage"))
        checks.append(Check(f"{b} short: |dE| / E", abs(z["E"] - a["E"]) / a["E"], 1e-5,
                            kind="max", source="relative kinetic energy -> heat; "
                                               "float storage"))
        a, z, last = res["long"]
        ac, zc, _ = res["control"]
        dE = abs(z["E"] - a["E"]) / a["E"]
        dEc = abs(zc["E"] - ac["E"]) / ac["E"]
        checks.append(Check(f"{b} long: mass", z["M"], a["M"], kind="rel", tol=1e-6,
                            source="float storage"))
        dP = np.abs(z["P"] - a["P"]).max() / a["Pscale"]
        dPc = np.abs(zc["P"] - ac["P"]).max() / ac["Pscale"]
        checks.append(Check(f"{b} long: |dP| / sum m|v|", dP, max(2 * dPc + 1e-5, 5e-4),
                            kind="max",
                            source=f"control run without merging: {dPc:.3g}; snapshot "
                                   "velocities of particles in mid-step are "
                                   "extrapolated with their accelerations, which "
                                   "moves sum m v by up to 3e-4 between snapshots "
                                   "of the MFM run without a trend"))
        checks.append(Check(f"{b} long: |dE| / E", dE, 2 * dEc + 1e-5, kind="max",
                            source=f"control run without merging: {dEc:.3g}"))
        light = last.read(0, "WindLight", cgs=False) > 0
        m = z["m"]
        checks.append(Check(f"{b} long: light particles below m_g / 2",
                            int((light & (m < 0.5 * m_g)).sum()), 0, kind="equal",
                            source="all clocks run out by 23 Myr"))
        ctx.info[f"{b}"] = {"n_merge_short": n_merge,
                            "dE_long": dE, "dE_control": dEc,
                            "max_mass_over_mg": float(m.max() / m_g),
                            "min_mass_over_mg": float(m.min() / m_g),
                            "n_light_end": int(light.sum()),
                            "n_gas_end": int(len(m))}
        checks.append(Check(f"{b} long: max mass / m_g", m.max() / m_g, 1.5, kind="max",
                            source="design Sec. 4.2: masses within [m_g/2, 3/2 m_g]"))
    return checks
