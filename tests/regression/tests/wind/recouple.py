"""Decoupled wind particles recouple at d_rec or t_rec and energy is conserved.

A periodic 4 kpc box of gas on a 32^3 lattice (n_H = 0.01 cm^-3, T = 1e5 K,
no cooling) in a constant field g = 1e-9 cm s^-2 along z. Sixteen particles
near the centre are flagged WindDecoupled = 1 in the ICs, as wind particles
just launched: twelve fast ones (300 km/s, random directions) and four slow
ones (40 km/s), all at T = 3e6 K. With d_rec = max(0.5 kpc, 2 kernel
supports at launch) = 0.56 kpc and t_rec = 3 Myr the fast ones recouple by
distance (after 1.9 Myr) and the slow ones by time (they have then travelled
0.12 kpc). The run lasts 6 Myr with dt_max = 0.01 Myr, below every CFL step,
with the time-step limiter on.

Checks:
  1. every particle recoupled, the fast ones by distance and the slow ones by
     time;
  2. by distance: d_rec <= d < d_rec + |v| dt_max, i.e. within one time-step
     of reaching d_rec;
  3. by time: t_rec <= t - t_launch < t_rec + dt_max;
  4. total energy E = sum m (v^2/2 + u - g (z - L/2)) at every snapshot
     against t = 0, relative to E(0), across the recouplings and the shocks
     the recoupled particles drive. The wind particles carry about 2 per cent
     of E. Tolerance 1e-5. With dt_max = 0.01 Myr the offset of SWIFT's
     output velocity at a synchronisation point (v_full + a dt/2, since
     get_integer_time_begin returns the previous step's start), of order
     dt dE_kin/dt, stays below it;
  5. the wind energy recorded at launch and at recoupling: internal energy
     unchanged, kinetic energy changed by no more than the work g d.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common

DESCRIPTION = "wind recoupling within one step of d_rec or t_rec; total energy conserved to 1e-5"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.NOSF_FLAGS + ["--limiter", "--sync"]

N = 32
KERNEL_GAMMA = 1.825742  # cubic spline in 3D (default kernel)
L = 4.0  # kpc
NH = 0.01
T_GAS = 1e5
G_CGS = 1e-9
N_FAST, N_SLOW = 12, 4
V_FAST, V_SLOW = 300.0, 40.0
T_WIND = 3e6
D_REC = 0.5  # kpc
T_REC = 3.0  # Myr
T_END = 6.0  # Myr
DT_MAX = 0.01  # Myr


def write_ics(path):
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pos)
    k = N_FAST + N_SLOW
    m = units.rho_from_nH(NH) * (L * units.KPC) ** 3 / n / units.UNIT_MASS
    rng = np.random.default_rng(7)
    wpos = 0.5 * L + rng.uniform(-0.5, 0.5, (k, 3))
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    speed = np.concatenate([np.full(N_FAST, V_FAST), np.full(N_SLOW, V_SLOW)])
    u = units.u_from_T(T_GAS, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    uw = units.u_from_T(T_WIND, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": np.vstack([pos, wpos]),
        "Velocities": np.vstack([np.zeros((n, 3)), d * speed[:, None]]),
        "Masses": np.full(n + k, m),
        "SmoothingLength": np.full(n + k, 1.2348 * L / N),
        "InternalEnergy": np.concatenate([np.full(n, u), np.full(k, uw)]),
        "ParticleIDs": np.arange(1, n + k + 1, dtype=np.uint64),
        "WindDecoupled": np.concatenate([np.zeros(n, np.int8), np.ones(k, np.int8)]),
    }
    ics.write(path, L, gas=gas)


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    write_ics(ctx.path("ics.hdf5"))
    myr = _common.MYR
    p = _common.base_params(ctx, "ics.hdf5", T_END * myr, 0.25 * myr, DT_MAX * myr,
                            g_cgs=(0.0, 0.0, G_CGS))
    p["Wind"] = {"loading_model": "none", "recoupling_distance_kpc": D_REC,
                 "recoupling_time_Myr": T_REC}
    ctx.swift(p, FLAGS, name="recouple", timeout=1800)


def energy(snap, g):
    m = snap.read(0, "Masses", cgs=False)
    v = snap.read(0, "Velocities", cgs=False)
    u = snap.read(0, "InternalEnergies", cgs=False)
    z = snap.read(0, "Coordinates", cgs=False)[:, 2]
    return np.sum(m * (0.5 * (v ** 2).sum(axis=1) + u - g * (z - 0.5 * L)))


def analyze(ctx):
    checks = []
    snaps = snapshots(ctx.run_dir)
    last = snaps[-1]
    myr = _common.MYR
    g = G_CGS / _common.ACC_UNIT
    n = N ** 3
    ids = last.read(0, "ParticleIDs", cgs=False).astype(np.int64)
    w = ids > n
    o = np.argsort(ids[w])
    fast = np.arange(N_FAST + N_SLOW) < N_FAST
    t_rec = last.read(0, "WindRecouplingTimes", cgs=False)[w][o]
    t_l = last.read(0, "WindLaunchTimes", cgs=False)[w][o]
    d_rec = last.read(0, "WindRecouplingDistances", cgs=False)[w][o]
    d_thr = last.read(0, "WindRecouplingDistanceThresholds", cgs=False)[w][o]
    ek = last.read(0, "WindRecouplingKineticEnergies", cgs=False)[w][o]
    eth = last.read(0, "WindRecouplingInternalEnergies", cgs=False)[w][o]
    ek0 = last.read(0, "WindLaunchKineticEnergies", cgs=False)[w][o]
    eth0 = last.read(0, "WindLaunchInternalEnergies", cgs=False)[w][o]
    dec = last.read(0, "WindDecoupled", cgs=False)[w][o]
    dt = DT_MAX * myr

    checks.append(Check("recoupled", int((dec == 0).sum()), N_FAST + N_SLOW,
                        kind="equal", source="all reach d_rec or t_rec in the run"))
    by_d = fast & (d_rec >= d_thr) & (t_rec - t_l < T_REC * myr)
    checks.append(Check("fast ones recoupled by distance", int(by_d.sum()), N_FAST,
                        kind="equal", source="d_rec / v = 1.6 Myr < t_rec"))
    vrec = np.sqrt(2 * ek[fast])
    excess = (d_rec[fast] - d_thr[fast]) / (vrec * dt)
    checks.append(Check("by distance: max (d - d_rec) / (|v| dt_max)", excess.max(), 1.0,
                        kind="max", source="checked at the end of every step"))
    checks.append(Check("d_rec = max(0.5 kpc, 2 kernel supports)", d_thr.min(),
                        max(D_REC, 2 * KERNEL_GAMMA * 1.2348 * L / N), kind="rel",
                        tol=1e-2, source="kernel support at launch: smoothing length "
                                         "of the ICs, cubic-spline gamma"))
    checks.append(Check("by distance: min (d - d_rec)", (d_rec[fast] - d_thr[fast]).min(), 0.0,
                        kind="min", source="not before d_rec"))
    tt = (t_rec[~fast] - t_l[~fast]) / myr - T_REC
    checks.append(Check("by time: min (t - t_launch - t_rec) [Myr]", tt.min(), -1e-6,
                        kind="min", source="not before t_rec (float launch time)"))
    checks.append(Check("by time: max (t - t_launch - t_rec) [Myr]", tt.max(),
                        DT_MAX * (1 + 1e-6),
                        kind="max", source="within one dt_max"))

    E = np.array([energy(s, g) for s in snaps])
    rel = np.abs(E / E[0] - 1)
    checks.append(Check("max |E(t) / E(0) - 1|", rel.max(), 1e-5, kind="max",
                        source="sum m (v^2/2 + u - g z), no cooling"))
    # While decoupled the internal energy is unchanged and the kinetic energy
    # changes only by the work of g over the displacement
    checks.append(Check("max |u_rec / u_launch - 1|", np.abs(eth / eth0 - 1).max(), 0.0,
                        kind="max", source="no hydro while decoupled"))
    # displacement at recoupling: z from ballistic motion with the launch
    # velocity, which the snapshot does not keep; use e_kin difference = g dz
    # with dz = v0z t + g t^2 / 2 bounded by |dz| <= d: check the identity
    # e_kin,rec - e_kin,launch <= g d (the work cannot exceed g times the path)
    work = ek - ek0
    checks.append(Check("max |e_kin,rec - e_kin,launch| / (g d)",
                        (np.abs(work) / (g * d_rec)).max(), 1.0 + 1e-3, kind="max",
                        source="only g acts while decoupled"))
    m = last.read(0, "Masses", cgs=False)[w][o]
    E_wind0 = np.sum(m * (ek0 + eth0))
    ctx.info.update({
        "E_rel_history": rel.tolist(), "E_wind_launch_over_E0": E_wind0 / E[0],
        "dE_max_over_E_wind": np.abs(E - E[0]).max() / E_wind0,
        "t_recouple_Myr": (t_rec / myr).tolist(), "d_recouple_kpc": d_rec.tolist()})
    return checks
