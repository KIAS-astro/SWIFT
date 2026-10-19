"""Warm wind particles recouple at their apex or at d_rec; CR energy is released
exactly and total energy including it is conserved (wind-design.md Sec. 10).

The 4 kpc periodic box of wind/recouple (32^3 lattice, n_H = 0.01 cm^-3,
T = 1e5 K, no cooling) in a constant field g = 2e-8 cm s^-2 along -z (the launch
direction n = +z points up). Particles decoupled by hand in the ICs
(WindDecoupled = 1, with WindPhases and WindCREnergies):
  - 8 slow warm particles (phase 2, T = 1e4 K, v_z = 40-70 km/s upwards plus a
    horizontal 30-60 km/s): apex at t = v_z / g (6-11 Myr), height
    v_z^2 / (2 g) (0.13-0.40 kpc), below d_rec,warm = 1.2 kpc, so they recouple
    at the apex;
  - 4 fast warm particles (v_z = 250-300 km/s): apex height > 5 kpc, so they
    recouple at d_rec,warm = 1.2 kpc;
  - 6 hot particles (phase 1, 300 km/s, random directions, T = 3e6 K) carrying
    E_CR = m x 1e16 erg/g each; they recouple at d_rec = max(0.5 kpc, 2 kernel
    supports) = 0.56 kpc and release E_CR.
Two runs: cr_deposit thermal and none. t_rec = 50 Myr, run 15 Myr, dt_max
0.01 Myr.

Validation checks (each run):
  1. slow warm particles: recoupling time against v_z / g (to dt_max) and
     displacement at recoupling against the analytic parabola
     |(v_h t, v_z t - g t^2 / 2)| at the recoupling time (relative 1e-4); every
     one recoupled by the apex test (log: apex recouplings = 8);
  2. fast warm particles: d_rec,warm <= d < d_rec,warm + |v| dt_max;
  3. hot particles: recoupled, CR energy 0 afterwards;
  4. CR bookkeeping from the log (doubles): launched = released (1e-12) =
     6 m x 1e16 erg/g (1e-6, float masses), deposited = released (thermal) or
     0 (none);
  5. energy in the frame falling with the field (v -> v + g t z, no potential:
     the periodic box wraps z): E(t) = sum m (v^2/2 + u) + sum E_CR,carried at
     every snapshot against t = 0, to 5e-5 of E(0) (thermal); with none,
     E(t) + released = E(0) to 1e-5. The CR energy is 8 per cent of E(0), so
     a missing or double deposit fails by 8e-2.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common
from . import _twophase as tp

DESCRIPTION = "warm wind recouples at its apex (analytic parabola); CR energy launched = deposited; E conserved"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.NOSF_FLAGS + ["--limiter", "--sync"]

N = 32
L = 4.0  # kpc
NH = 0.01
T_GAS = 1e5
G_CGS = 2e-8  # along -z
N_SLOW, N_FAST, N_HOT = 8, 4, 6
T_WARM, T_HOT = 1e4, 3e6
E_CR_PER_MASS = 1e16  # erg / g
D_REC, D_REC_WARM = 0.5, 1.2  # kpc
T_REC = 50.0  # Myr
T_END = 15.0  # Myr
DT_MAX = 0.01  # Myr
RUNS = ("thermal", "none")


def wind_set():
    rng = np.random.default_rng(11)
    v = []
    for i in range(N_SLOW):
        vz = 40.0 + 30.0 * i / (N_SLOW - 1)
        vh = 30.0 + 30.0 * rng.uniform()
        a = rng.uniform(0, 2 * np.pi)
        v.append([vh * np.cos(a), vh * np.sin(a), vz])
    for i in range(N_FAST):
        a = rng.uniform(0, 2 * np.pi)
        v.append([20.0 * np.cos(a), 20.0 * np.sin(a), 250.0 + 50.0 * i / (N_FAST - 1)])
    d = rng.normal(size=(N_HOT, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    v.extend((300.0 * d).tolist())
    pos = 0.5 * L + rng.uniform(-0.3, 0.3, (len(v), 3))
    return pos, np.array(v) * 1e5 / units.UNIT_VELOCITY


def write_ics(path):
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pos)
    wpos, wv = wind_set()
    k = len(wpos)
    m = units.rho_from_nH(NH) * (L * units.KPC) ** 3 / n / units.UNIT_MASS
    u = units.u_from_T(T_GAS, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    uw = units.u_from_T(T_WARM, 0.6) / units.UNIT_ENERGY_PER_MASS
    uh = units.u_from_T(T_HOT, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    nwarm = N_SLOW + N_FAST
    phase = np.concatenate([np.zeros(n, np.int8), np.full(nwarm, 2, np.int8),
                            np.ones(N_HOT, np.int8)])
    e_cr = E_CR_PER_MASS / units.UNIT_ENERGY_PER_MASS * m
    gas = {
        "Coordinates": np.vstack([pos, wpos]),
        "Velocities": np.vstack([np.zeros((n, 3)), wv]),
        "Masses": np.full(n + k, m),
        "SmoothingLength": np.full(n + k, 1.2348 * L / N),
        "InternalEnergy": np.concatenate([np.full(n, u), np.full(nwarm, uw),
                                          np.full(N_HOT, uh)]),
        "ParticleIDs": np.arange(1, n + k + 1, dtype=np.uint64),
        "WindDecoupled": np.concatenate([np.zeros(n, np.int8), np.ones(k, np.int8)]),
        "WindPhases": phase,
        "WindCREnergies": np.concatenate([np.zeros(n + nwarm), np.full(N_HOT, e_cr)]),
    }
    ics.write(path, L, gas=gas)


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    myr = _common.MYR
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        write_ics(ctx.path(name, "ics.hdf5"))
        p = _common.base_params(ctx, "ics.hdf5", T_END * myr, 0.25 * myr, DT_MAX * myr,
                                g_cgs=(0.0, 0.0, -G_CGS))
        p["Wind"] = {"loading_model": "none", "recoupling_distance_kpc": D_REC,
                     "recoupling_time_Myr": T_REC,
                     "warm_recoupling_distance_kpc": D_REC_WARM,
                     "cr_deposit": name}
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=1800)


def energy(snap, g):
    """Energy in the frame falling with the uniform field, where no external
    force acts: the periodic box wraps z, so a potential term g z jumps by g L
    when the free-falling gas crosses the boundary (0.7 kpc of fall in 15 Myr)."""
    m = snap.read(0, "Masses", cgs=False)
    v = snap.read(0, "Velocities", cgs=False).copy()
    u = snap.read(0, "InternalEnergies", cgs=False)
    ecr = snap.read(0, "WindCREnergies", cgs=False)
    v[:, 2] += g * snap.time_cgs / units.UNIT_TIME
    return np.sum(m * (0.5 * (v ** 2).sum(axis=1) + u)), ecr.sum()


def analyze(ctx):
    checks = []
    myr = _common.MYR
    g = G_CGS / _common.ACC_UNIT
    n = N ** 3
    _, wv = wind_set()
    vz0, vh0 = wv[:N_SLOW, 2], np.hypot(wv[:N_SLOW, 0], wv[:N_SLOW, 1])
    dt = DT_MAX * myr
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        ids = last.read(0, "ParticleIDs", cgs=False).astype(np.int64)
        w = ids > n
        o = np.argsort(ids[w])
        rd = lambda f: last.read(0, f, cgs=False)[w][o]
        t_rec, t_l = rd("WindRecouplingTimes"), rd("WindLaunchTimes")
        d_rec, d_thr = rd("WindRecouplingDistances"), rd("WindRecouplingDistanceThresholds")
        dec, ecr = rd("WindDecoupled"), rd("WindCREnergies")
        ek = rd("WindRecouplingKineticEnergies")

        slow = slice(0, N_SLOW)
        fast = slice(N_SLOW, N_SLOW + N_FAST)
        hot = slice(N_SLOW + N_FAST, None)
        checks.append(Check(f"{name}: all recoupled", int((dec == 0).sum()),
                            N_SLOW + N_FAST + N_HOT, kind="equal",
                            source="apex, d_rec,warm or d_rec within the run"))
        t_apex = vz0 / g
        tt = t_rec[slow] - t_l[slow]
        checks.append(Check(f"{name}: slow warm: max |t_rec - v_z/g| / dt_max",
                            np.abs(tt - t_apex).max() / dt, 1.0 + 1e-6, kind="max",
                            source="apex tested at the end of every step"))
        d_an = np.hypot(vh0 * tt, vz0 * tt - 0.5 * g * tt ** 2)
        checks.append(Check(f"{name}: slow warm: max |d_rec / parabola - 1|",
                            np.abs(d_rec[slow] / d_an - 1).max(), 1e-4, kind="max",
                            source="ballistic in the constant field, at the recoupling time"))
        vrec = np.sqrt(2 * ek[fast])
        exc = (d_rec[fast] - d_thr[fast]) / (vrec * dt)
        checks.append(Check(f"{name}: fast warm: d_rec,warm threshold",
                            float(d_thr[fast].min()), D_REC_WARM, kind="rel", tol=1e-6,
                            source="Wind:warm_recoupling_distance_kpc"))
        checks.append(Check(f"{name}: fast warm: max (d - d_rec) / (|v| dt_max)",
                            exc.max(), 1.0, kind="max", source="within one step"))
        checks.append(Check(f"{name}: fast warm: min (d - d_rec)",
                            (d_rec[fast] - d_thr[fast]).min(), 0.0, kind="min",
                            source="not before d_rec"))
        checks.append(Check(f"{name}: hot: CR energy left", float(np.abs(ecr[hot]).sum()), 0.0,
                            kind="abs", tol=0.0, source="released at recoupling"))
        tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
        checks.append(Check(f"{name}: apex recouplings (log)", tail["n_apex"], N_SLOW,
                            kind="equal", source="slow warm only"))
        m0 = rd("Masses")
        E_cr0 = float(np.sum(m0[hot])) * E_CR_PER_MASS / units.UNIT_ENERGY_PER_MASS
        checks.append(Check(f"{name}: CR launched (log) / 6 m e_CR", tail["E_cr_launch"] / E_cr0,
                            1.0, kind="rel", tol=1e-6, source="ICs, float masses"))
        checks.append(Check(f"{name}: CR released / launched (log)",
                            tail["E_cr_release"] / tail["E_cr_launch"], 1.0, kind="rel",
                            tol=1e-12, source="double sums of the same per-particle doubles"))
        dep_exp = tail["E_cr_release"] if name == "thermal" else 0.0
        checks.append(Check(f"{name}: CR deposited (log)", tail["E_cr_deposit"], dep_exp,
                            kind="rel" if dep_exp else "abs", tol=1e-12 if dep_exp else 0.0,
                            source=f"cr_deposit: {name}"))
        E = np.array([energy(s, g) for s in snaps])
        Etot0 = E[0, 0] + E[0, 1]
        # Tolerance: 1e-5 as wind/recouple; with the thermal deposit 5e-5,
        # measured 2.3e-5 (3e-4 of the deposited energy, which drives six
        # blasts at 4e7 K), against 8e-2 for a missing or doubled deposit
        if name == "thermal":
            Et = E[:, 0] + E[:, 1]
            src = "falling frame: gas + carried CR, no cooling"
            tol = 5e-5
        else:
            Et = E[:, 0] + E[0, 1]
            src = "falling frame: gas + carried CR + released (discarded) CR"
            tol = 1e-5
        rel = np.abs(Et / Etot0 - 1)
        checks.append(Check(f"{name}: max |E(t) / E(0) - 1|", rel.max(), tol, kind="max",
                            source=src))
        ctx.info[name] = {"E_rel": rel.tolist(), "E_cr_over_E0": E[0, 1] / Etot0,
                          "t_apex_Myr": (t_apex / myr).tolist(),
                          "t_rec_slow_Myr": (tt / myr).tolist(), "log": tail}
    return checks
