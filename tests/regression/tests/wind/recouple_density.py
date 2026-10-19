"""Recoupling by ambient density, and deposit of the wind particle into a host.

A periodic 4 kpc box: a dense slab |z - 2 kpc| < 1 kpc on a lattice of spacing
0.125 kpc (n_H = 0.08 cm^-3, T = 2e4 K) between dilute layers of spacing 0.25 kpc
(n_H = 0.01, T = 1.6e5 K; equal pressure, equal particle masses), no cooling and no
gravity. Eight particles at the slab mid-plane are flagged WindDecoupled = 1 in the
ICs, as wind particles just launched along +-z at 300 km/s and T = 3e6 K, with
Wind:recoupling_mode density and a threshold n_H = 0.024 cm^-3 (0.3 of the slab).
The ambient density is the scatter estimate sum_j m_j W(|x - x_j|, h_j) over the
non-decoupled gas, computed at the start of every step (src/wind.c).

Two runs from the same ICs: deposit 'self' (the particle rejoins the hydro) and
deposit 'host' (merged into its nearest gas neighbour and removed).

Checks, run 'self':
  1. every wind particle recoupled, all by density (WindRecouplingReasons = 2);
  2. not inside the slab: the recoupling distance from the mid-plane is >= 1 kpc
     minus one time-step of travel. Inside the slab the scatter density is at least
     half the slab value (0.04 > 0.024) up to the interface;
  3. within the transition: distance <= 1 kpc + one kernel support of the dilute
     gas (1.825 x 1.2348 x 0.25 = 0.56 kpc) + one step of travel, since beyond that
     only dilute particles contribute (0.01 < 0.024);
  4. the ambient density recorded at recoupling is below the threshold.
Checks, run 'host':
  5. eight gas particles fewer at the end, and the log counts eight merges;
  6. total mass of gas at the end equals the start to 1e-6 (float32 masses);
  7. total momentum sum m v (zero by symmetry at the start) stays below 1e-4 of the
     wind particles' sum m |v|: the merge conserves momentum exactly, SPH to round-off;
  8. total energy sum m (v^2/2 + u) at every snapshot equals the start to 5e-5 (the
     merge turns the relative kinetic energy into heat), in both runs. The 'self'
     run, with no merge, shows excursions of up to 2.9e-5 at snapshots between
     synchronisation points: SWIFT writes v_full + a dt/2 there (wind/recouple), and
     the slab interface, not in exact equilibrium, has accelerations larger than the
     uniform gas of wind/recouple, where the same offset stays below 1e-5. The
     merges change the 'host' run's energy history by under 1e-5 from 'self'.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common

DESCRIPTION = "wind recoupling by ambient density; merge into a host conserves mass, momentum, energy"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.NOSF_FLAGS + ["--limiter", "--sync"]

L = 4.0  # kpc
A_DENSE, A_DILUTE = 0.125, 0.25  # kpc
NH_DENSE = 0.08
T_DENSE = 2e4
NH_REC = 0.024
N_WIND = 8
V_WIND = 300.0  # km/s
T_WIND = 3e6
T_END = 6.0  # Myr
DT_MAX = 0.01  # Myr
KERNEL_GAMMA = 1.825742


def lattice(a, zlo, zhi):
    x = np.arange(0.5 * a, L, a)
    z = np.arange(zlo + 0.5 * a, zhi, a)
    xx, yy, zz = np.meshgrid(x, x, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def write_ics(path):
    dense = lattice(A_DENSE, 1.0, 3.0)
    dilute = np.vstack([lattice(A_DILUTE, 0.0, 1.0), lattice(A_DILUTE, 3.0, 4.0)])
    m = units.rho_from_nH(NH_DENSE) * (A_DENSE * units.KPC) ** 3 / units.UNIT_MASS
    ratio = (A_DILUTE / A_DENSE) ** 3
    u_d = units.u_from_T(T_DENSE, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    u_l = u_d * ratio
    u_w = units.u_from_T(T_WIND, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    rng = np.random.default_rng(3)
    wpos = np.column_stack([rng.uniform(1.0, 3.0, N_WIND), rng.uniform(1.0, 3.0, N_WIND),
                            np.full(N_WIND, 0.5 * L)])
    sign = np.where(np.arange(N_WIND) % 2 == 0, 1.0, -1.0)
    wvel = np.column_stack([np.zeros(N_WIND), np.zeros(N_WIND), sign * V_WIND])
    n_d, n_l = len(dense), len(dilute)
    gas = {
        "Coordinates": np.vstack([dense, dilute, wpos]),
        "Velocities": np.vstack([np.zeros((n_d + n_l, 3)), wvel]),
        "Masses": np.full(n_d + n_l + N_WIND, m),
        "SmoothingLength": np.concatenate([np.full(n_d, 1.2348 * A_DENSE),
                                           np.full(n_l, 1.2348 * A_DILUTE),
                                           np.full(N_WIND, 1.2348 * A_DENSE)]),
        "InternalEnergy": np.concatenate([np.full(n_d, u_d), np.full(n_l, u_l),
                                          np.full(N_WIND, u_w)]),
        "ParticleIDs": np.arange(1, n_d + n_l + N_WIND + 1, dtype=np.uint64),
        "WindDecoupled": np.concatenate([np.zeros(n_d + n_l, np.int8),
                                         np.ones(N_WIND, np.int8)]),
    }
    ics.write(path, L, gas=gas)
    return n_d + n_l


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    myr = _common.MYR
    for deposit in ("self", "host"):
        os.makedirs(ctx.path(deposit))
        write_ics(ctx.path(deposit, "ics.hdf5"))
        p = _common.base_params(ctx, "ics.hdf5", T_END * myr, 0.5 * myr, DT_MAX * myr)
        p["Wind"] = {"loading_model": "none", "recoupling_mode": "density",
                     "recoupling_density_H_p_cm3": NH_REC,
                     "recoupling_deposit": deposit, "recoupling_time_Myr": 100.0}
        ctx.swift(p, FLAGS, subdir=deposit, name=deposit, timeout=1800)


def totals(snap):
    m = snap.read(0, "Masses", cgs=False)
    v = snap.read(0, "Velocities", cgs=False)
    u = snap.read(0, "InternalEnergies", cgs=False)
    return m.sum(), (m[:, None] * v).sum(axis=0), np.sum(m * (0.5 * (v ** 2).sum(axis=1) + u))


def analyze(ctx):
    checks = []
    myr = _common.MYR
    n_gas = (int(L / A_DENSE) ** 2 * int(2.0 / A_DENSE) + int(L / A_DILUTE) ** 2 * int(2.0 / A_DILUTE))

    last = snapshots(ctx.path("self"))[-1]
    ids = last.read(0, "ParticleIDs", cgs=False).astype(np.int64)
    w = ids > n_gas
    dec = last.read(0, "WindDecoupled", cgs=False)[w]
    reason = last.read(0, "WindRecouplingReasons", cgs=False)[w]
    d = last.read(0, "WindRecouplingDistances", cgs=False)[w]
    rho = last.read(0, "WindAmbientDensities")[w]
    step = V_WIND * DT_MAX * myr  # kpc per dt_max (internal velocity unit km/s)
    checks.append(Check("self: recoupled", int((dec == 0).sum()), N_WIND, kind="equal",
                        source="all leave the slab"))
    checks.append(Check("self: by density", int((reason == 2).sum()), N_WIND, kind="equal",
                        source="WindRecouplingReasons = 2"))
    checks.append(Check("self: min recoupling distance [kpc]", d.min(), 1.0 - step,
                        kind="min", source="scatter density >= 0.04 inside the slab"))
    checks.append(Check("self: max recoupling distance [kpc]", d.max(),
                        1.0 + KERNEL_GAMMA * 1.2348 * A_DILUTE + step, kind="max",
                        source="beyond one dilute kernel support only dilute gas"))
    rho_rec = units.rho_from_nH(NH_REC)
    checks.append(Check("self: max ambient density at recoupling / threshold",
                        rho.max() / rho_rec, 1.0, kind="max", source="criterion"))

    snaps = snapshots(ctx.path("host"))
    first, last = snaps[0], snaps[-1]
    n0, n1 = len(first.read(0, "Masses", cgs=False)), len(last.read(0, "Masses", cgs=False))
    checks.append(Check("host: gas particles removed", n0 - n1, N_WIND, kind="equal",
                        source="each wind particle merged into a host"))
    tail = _common.wind_tail(ctx.path("host", "host.log"))
    merged = -1
    with open(ctx.path("host", "host.log"), errors="replace") as f:
        for line in f:
            if "merged into a host" in line:
                merged = int(float(line.rsplit(" ", 1)[1]))
    checks.append(Check("host: merges in the log", merged, N_WIND, kind="equal",
                        source="Wind recouplings by criterion line"))
    M0, P0, E0 = totals(first)
    m_w = first.read(0, "Masses", cgs=False)[-N_WIND:]
    p_scale = np.sum(m_w * V_WIND)
    rel_E, rel_P = [], []
    for s in snaps:
        M, P, E = totals(s)
        rel_E.append(abs(E / E0 - 1))
        rel_P.append(np.linalg.norm(P - P0) / p_scale)
    M1 = totals(last)[0]
    checks.append(Check("host: total mass end / start", M1 / M0, 1.0, kind="rel", tol=1e-6,
                        source="merge moves mass"))
    checks.append(Check("host: max |P - P0| / sum m_w |v_w|", max(rel_P), 1e-4, kind="max",
                        source="merge and SPH conserve momentum"))
    checks.append(Check("host: max |E / E0 - 1|", max(rel_E), 5e-5, kind="max",
                        source="sum m (v^2/2 + u), no cooling or gravity"))
    s_snaps = snapshots(ctx.path("self"))
    E0s = totals(s_snaps[0])[2]
    rel_Es = [abs(totals(s)[2] / E0s - 1) for s in s_snaps]
    checks.append(Check("self: max |E / E0 - 1|", max(rel_Es), 5e-5, kind="max",
                        source="as above, no merge"))
    ctx.info.update({"self_recouple_distance_kpc": d.tolist(),
                     "host_rel_E": rel_E, "host_rel_P": rel_P, "log_tail": tail})
    return checks
