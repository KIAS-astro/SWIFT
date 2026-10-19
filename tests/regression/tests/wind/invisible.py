"""Decoupled wind particles are invisible to the gas and move ballistically.

A periodic 1 kpc box of gas on a 16^3 lattice (n_H = 0.05 cm^-3, below the
PRFM floor threshold, T = 1e4 K) carrying a sound wave v_x = A sin(2 pi x / L),
A = 1 km/s, in a constant field g = 1e-8 cm s^-2 along z, no cooling. Three
runs of 2 Myr from the same ICs:
  B  the gas alone;
  A  the gas plus 24 hot (3e6 K), fast (100-300 km/s) particles of the gas
     particle mass, flagged WindDecoupled = 1 in the ICs;
  C  as A with the flag 0 (a control: the extra particles are ordinary gas).
dt_max = 2 Myr / 128 is below every CFL and gravity limit, so all particles
share one time-step and the runs take the same steps.

Checks:
  1-3. densities, velocities and internal energies of every gas particle in
       A against B. The only differences allowed are those of round-off, from
       the different cell tree and order of summation: relative 1e-5 for
       density and internal energy, 1e-4 A for velocity;
  4.   control: in C the same differences exceed 1e-3 (one neighbour of the
       ~50 in a kernel changes the density by ~2e-2), so checks 1-3 would
       fail if the wind particles were seen;
  5.   each decoupled particle's displacement (accumulated in the drift,
       unwrapped) against v0 t + g t^2 / 2. Leapfrog is exact in a constant
       field; what is left is the rounding of the float velocity v_full at
       each kick, N_step 2^-24 = 7.6e-6 relative. (The snapshot velocities
       are not used: for every gas particle, coupled or not, SWIFT's output
       velocity in this set-up is ahead by g dt, an output convention that
       the positions do not share.)
  7.   their internal energy unchanged (no hydro, no cooling, no floor).
"""

import os

import h5py
import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common

DESCRIPTION = "decoupled particles: gas identical to a run without them; ballistic in a constant field"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}

N = 16
L = 1.0  # kpc
NH = 0.05
T_GAS = 1e4
A_WAVE = 1.0  # km/s
G_CGS = 1e-8
K = 24
T_WIND = 3e6
T_END = 2.0  # Myr
NSTEP = 128


def write_ics(path, wind, flag):
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pos)
    m = units.rho_from_nH(NH) * (L * units.KPC) ** 3 / n / units.UNIT_MASS
    vel = np.zeros((n, 3))
    vel[:, 0] = A_WAVE * np.sin(2 * np.pi * pos[:, 0] / L)
    u = np.full(n, units.u_from_T(T_GAS, units.MU_NEUTRAL) / units.UNIT_ENERGY_PER_MASS)
    gas = {"Coordinates": pos, "Velocities": vel, "Masses": np.full(n, m),
           "SmoothingLength": np.full(n, 1.2348 * L / N), "InternalEnergy": u,
           "ParticleIDs": np.arange(1, n + 1, dtype=np.uint64)}
    if wind:
        rng = np.random.default_rng(42)
        wpos = rng.uniform(0, L, (K, 3))
        d = rng.normal(size=(K, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        wvel = d * rng.uniform(100.0, 300.0, K)[:, None]
        uw = units.u_from_T(T_WIND, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
        gas = {
            "Coordinates": np.vstack([pos, wpos]),
            "Velocities": np.vstack([vel, wvel]),
            "Masses": np.full(n + K, m),
            "SmoothingLength": np.full(n + K, 1.2348 * L / N),
            "InternalEnergy": np.concatenate([u, np.full(K, uw)]),
            "ParticleIDs": np.arange(1, n + K + 1, dtype=np.uint64),
            "WindDecoupled": np.concatenate([np.zeros(n, np.int8),
                                             np.full(K, flag, np.int8)]),
        }
    ics.write(path, L, gas=gas)


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    t_end = T_END * _common.MYR
    for name, wind, flag in (("B", False, 0), ("A", True, 1), ("C", True, 0)):
        os.makedirs(ctx.path(name))
        write_ics(ctx.path(name, "ics.hdf5"), wind, flag)
        p = _common.base_params(ctx, "ics.hdf5", t_end, t_end, t_end / NSTEP,
                                g_cgs=(0.0, 0.0, G_CGS))
        p["Wind"] = {"loading_model": "none", "recoupling_distance_kpc": 1e3,
                     "recoupling_time_Myr": 1e4}
        ctx.swift(p, _common.NOSF_FLAGS, subdir=name, name=name, timeout=1800)


def _by_id(snap, name):
    ids = snap.read(0, "ParticleIDs", cgs=False).astype(np.int64)
    data = snap.read(0, name, cgs=False)
    out = np.empty_like(data)
    out[ids - 1] = data
    return out


def analyze(ctx):
    checks = []
    s = {k: snapshots(ctx.path(k))[-1] for k in "ABC"}
    n = N ** 3
    diff = {}
    for k in "AC":
        rho = [_by_id(s[x], "Densities")[:n] for x in (k, "B")]
        vel = [_by_id(s[x], "Velocities")[:n] for x in (k, "B")]
        u = [_by_id(s[x], "InternalEnergies")[:n] for x in (k, "B")]
        diff[k] = (np.abs(rho[0] / rho[1] - 1).max(),
                   np.abs(vel[0] - vel[1]).max() / A_WAVE,
                   np.abs(u[0] / u[1] - 1).max())
    checks.append(Check("A vs B: max |rho / rho_B - 1|", diff["A"][0], 1e-5, kind="max",
                        source="round-off of the summation order"))
    checks.append(Check("A vs B: max |v - v_B| / A", diff["A"][1], 1e-4, kind="max",
                        source="round-off of the summation order"))
    checks.append(Check("A vs B: max |u / u_B - 1|", diff["A"][2], 1e-5, kind="max",
                        source="round-off of the summation order"))
    checks.append(Check("control C vs B: max |rho / rho_B - 1|", diff["C"][0], 1e-3,
                        kind="min", source="coupled extra particles are seen"))

    a = s["A"]
    t = a.time
    g = G_CGS / _common.ACC_UNIT
    ids = a.read(0, "ParticleIDs", cgs=False).astype(np.int64)
    w = ids > n
    order = np.argsort(ids[w])
    with h5py.File(ctx.path("A", "ics.hdf5"), "r") as f:
        v0 = f["PartType0/Velocities"][n:].astype(np.float32).astype(np.float64)
        u0 = f["PartType0/InternalEnergy"][n:].astype(np.float32).astype(np.float64)
    acc = np.array([0.0, 0.0, g])
    dx_exp = v0 * t + 0.5 * acc * t * t
    dx = a.read(0, "WindDisplacements", cgs=False)[w][order]
    u = a.read(0, "InternalEnergies", cgs=False)[w][order]
    err_x = (np.linalg.norm(dx - dx_exp, axis=1) / np.linalg.norm(dx_exp, axis=1)).max()
    checks.append(Check("wind: max |dx - (v0 t + g t^2/2)| / |dx|", err_x,
                        NSTEP * 2.0 ** -24, kind="max",
                        source="leapfrog is exact in a constant field; v_full is "
                               "float, rounded once per kick: N_step 2^-24"))
    checks.append(Check("wind: max |u / u_0 - 1|", np.abs(u / u0 - 1).max(), 0.0,
                        kind="max", source="no hydro, cooling or floor"))
    dec = a.read(0, "WindDecoupled", cgs=False)[w]
    checks.append(Check("wind: still decoupled", int(dec.sum()), K, kind="equal",
                        source="recoupling beyond the run"))
    ctx.info.update({"A_vs_B": diff["A"], "C_vs_B": diff["C"], "t_Myr": t / _common.MYR,
                     "g_internal": g})
    return checks
