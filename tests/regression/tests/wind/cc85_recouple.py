"""Pressure-balance recoupling of decoupled wind particles from a central
CC85 source in a uniform medium (validation V3 with Wind:recoupling =
pressure; wind-design.md Sec. 9.2 and 9.4).

A periodic 20 kpc box of gas on a 32^3 lattice at n_H = 1e-4 cm^-3 and
T = 1e6 K (P/k = 2.25e2 K cm^-3), no cooling, no gravity. 64 particles are
decoupled in the ICs within 5 pc of the centre and move radially outwards at
500 km/s with an internal energy that makes their Bernoulli speed
(v^2 + 2 gamma u)^1/2 = v_inf = 1000 km/s. They carry a CC85 source of
radius R = 0.3 kpc and a mass rate Mdot of 0.1, 0.3, 1 or 3 Msun/yr, read
from the IC fields WindCC85Radii and WindCC85MassRates; one run per Mdot,
16 particles each, to (r_eq + 0.5 kpc) / 500 km/s. One run per Mdot because
the recoupled particles (u 19 times the ambient's) keep moving outwards at
almost their launch speed in gas this thin and raise the volume-weighted
shell pressure ahead of a slower group; within one run all 16 recouple in
the same step, before the profile is rebuilt. A
particle recouples when the CC85 pressure at its distance from its launch
point falls to the ambient pressure, which is the volume-weighted pressure
of the coupled gas in its shell about the box centre (Wind:ambient_centre).
d_min = 0.5 kpc, backstops d_max = 9 kpc and t_rec = 100 Myr (not reached).
dt_max = 0.01 Myr, 5 pc of travel per step.

Expected: the analytic balance radius r_eq of P_CC85(r_eq) = P_0, from
tests/wind/_cc85.py (independent of the C code), with P_0 the IC pressure
(gamma - 1) rho u: r_eq = 1.4-4.0 kpc.

Checks:
  1. all 64 recoupled, all by pressure (reason 3);
  2. per particle, recoupling distance - r_eq between -0.005 kpc (the
     launch offset from the centre) and the travel in one step plus the
     offset, v dt_max + 0.005 kpc = 0.0101 kpc;
  3. the ambient pressure recorded at recoupling against P_0, relative
     1e-2 (SPH density noise of a lattice and float storage);
  4. the CC85 pressure recorded at recoupling: at most the ambient pressure
     recorded with it, and at least P_CC85(r_eq + v dt_max + 0.005 kpc) (the
     trigger fires within one step).
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _cc85, _common

DESCRIPTION = "pressure recoupling: CC85 balance radius in a uniform medium (V3)"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.NOSF_FLAGS + ["--limiter", "--sync"]

N = 32
L = 20.0  # kpc
NH = 1e-4
T_GAS = 1e6
N_PER = 16
MDOTS = (0.1, 0.3, 1.0, 3.0)  # Msun / yr
R_CC85 = 0.3  # kpc
V0 = 500.0  # km/s
V_INF = 1000.0  # km/s
OFFSET = 0.005  # kpc
D_MIN = 0.5  # kpc
D_MAX = 9.0
T_REC = 100.0  # Myr
T_END = 10.0  # Myr
DT_MAX = 0.01  # Myr
MSUN_PER_YR = units.SOLAR_MASS / units.YEAR / (units.UNIT_MASS / units.UNIT_TIME)


def ambient_state():
    rho = units.rho_from_nH(NH) / units.UNIT_DENSITY
    u = units.u_from_T(T_GAS, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    return rho, u, (units.GAMMA - 1) * rho * u


def write_ics(path, mdot):
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pos)
    k = N_PER
    rho, u, _ = ambient_state()
    m = rho * L ** 3 / n
    rng = np.random.default_rng(11)
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    # Lattice points sit at +-L/(2N) from the centre, so the wind particles do
    # not coincide with gas particles
    wpos = 0.5 * L + OFFSET * d
    uw = (V_INF ** 2 - V0 ** 2) / (2 * units.GAMMA)
    gas = {
        "Coordinates": np.vstack([pos, wpos]),
        "Velocities": np.vstack([np.zeros((n, 3)), V0 * d]),
        "Masses": np.full(n + k, m),
        "SmoothingLength": np.concatenate([np.full(n, 1.2348 * L / N), np.full(k, 0.05)]),
        "InternalEnergy": np.concatenate([np.full(n, u), np.full(k, uw)]),
        "ParticleIDs": np.arange(1, n + k + 1, dtype=np.uint64),
        "WindDecoupled": np.concatenate([np.zeros(n, np.int8), np.ones(k, np.int8)]),
        "WindCC85Radii": np.concatenate([np.zeros(n), np.full(k, R_CC85)]).astype(np.float32),
        "WindCC85MassRates": np.concatenate(
            [np.zeros(n), np.full(k, mdot * MSUN_PER_YR)]).astype(np.float32),
    }
    ics.write(path, L, gas=gas)


def subdir(mdot):
    return f"mdot{mdot:g}"


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    myr = _common.MYR
    _, _, P0 = ambient_state()
    for mdot in MDOTS:
        sd = subdir(mdot)
        os.makedirs(ctx.path(sd))
        write_ics(ctx.path(sd, "ics.hdf5"), mdot)
        r_eq = _cc85.balance_radius(R_CC85, mdot * MSUN_PER_YR, V_INF, P0)
        t_end = (r_eq + 0.5) / V0
        p = _common.base_params(ctx, "ics.hdf5", t_end, 0.5 * t_end, DT_MAX * myr)
        p["Wind"] = {"loading_model": "none", "recoupling": "pressure",
                     "recoupling_distance_kpc": D_MAX, "recoupling_time_Myr": T_REC,
                     "recoupling_min_distance_kpc": D_MIN,
                     "ambient_centre": [0.0, 0.0, 0.0], "ambient_max_radius_kpc": 30.0,
                     "ambient_max_nH_cm3": 1.0}
        ctx.swift(p, FLAGS, name="cc85", subdir=sd, timeout=1800)


def analyze(ctx):
    checks = []
    n = N ** 3
    k = N_PER * len(MDOTS)
    cols = {name: [] for name in ("WindRecouplingReasons", "WindRecouplingDistances",
                                  "WindRecouplingAmbientPressures",
                                  "WindRecouplingWindPressures")}
    for mdot in MDOTS:
        last = snapshots(ctx.path(subdir(mdot)))[-1]
        ids = last.read(0, "ParticleIDs", cgs=False).astype(np.int64)
        w = ids > n
        o = np.argsort(ids[w])
        for name in cols:
            cols[name].append(last.read(0, name, cgs=False)[w][o])
    reason, d, P_amb, P_w = (np.concatenate(cols[c]) for c in cols)
    _, _, P0 = ambient_state()
    mdot = np.repeat(MDOTS, N_PER) * MSUN_PER_YR
    r_eq = np.array([_cc85.balance_radius(R_CC85, md * MSUN_PER_YR, V_INF, P0)
                     for md in MDOTS])
    r_eq_p = np.repeat(r_eq, N_PER)
    step = V0 * DT_MAX * _common.MYR

    checks.append(Check("recoupled by pressure", int((reason == 3).sum()), k,
                        kind="equal", source="P_CC85 falls to P_0 before d_max and t_rec"))
    dd = d - r_eq_p
    checks.append(Check("min (d - r_eq) [kpc]", dd.min(), -OFFSET - 1e-4, kind="min",
                        source="not before the balance radius, less the launch offset"))
    checks.append(Check("max (d - r_eq) [kpc]", dd.max(), step + OFFSET + 1e-4, kind="max",
                        source=f"within one step ({step:.4f} kpc) plus the launch offset"))
    checks.append(Check("max |P_amb / P_0 - 1| at recoupling",
                        np.abs(P_amb / P0 - 1).max(), 1e-2, kind="max",
                        source="undisturbed uniform medium, volume-weighted shell mean"))
    checks.append(Check("max P_w / P_amb at recoupling", (P_w / P_amb).max(), 1.0 + 1e-6,
                        kind="max", source="trigger P_w <= P_amb (float storage)"))
    P_next = np.array([_cc85.pressure(r + step + OFFSET, R_CC85, md, V_INF)
                       for r, md in zip(r_eq_p, mdot)])
    checks.append(Check("min P_w / P_CC85(r_eq + one step)", (P_w / P_next).min(),
                        1.0 - 1e-5, kind="min", source="fires within one step"))
    ctx.info.update({
        "r_eq_kpc": r_eq.tolist(), "P0_internal": P0,
        "d_recouple_kpc_by_mdot": [d[i * N_PER:(i + 1) * N_PER].tolist()
                                   for i in range(len(MDOTS))],
        "P_amb_over_P0": (P_amb / P0).tolist(), "P_w_over_P0": (P_w / P0).tolist()})
    return checks
