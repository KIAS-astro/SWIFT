"""Wind carriers deposit at d_rec or t_rec, and record which criterion fired.

The box of wind/recouple (periodic 4 kpc, 32^3 gas lattice, n_H = 0.01
cm^-3, T = 1e5 K, no cooling, g = 1e-9 cm s^-2 along z) with sixteen star
particles flagged WindCarrier = 1 in the ICs instead of decoupled gas
particles: twelve fast ones (300 km/s, random directions) and four slow ones
(40 km/s), mass m_g / 8, T_c = 3e6 K. d_rec = max(0.5 kpc, 2 kernel
supports of the IC smoothing length) = 0.56 kpc and t_rec = 3 Myr: the fast
ones reach d_rec after 1.8 Myr, the slow ones have travelled 0.12 kpc at
t_rec. dt_max = 0.01 Myr; 4 Myr.

The criterion is decided in the drift of the carrier (distance accumulated
there) and acted on at the end of the carrier's next active step.

Checks (from wind_carriers.txt, one line per deposit):
  1. sixteen deposits, no carrier left at the end, none retried (every
     carrier found gas in its kernel);
  2. the twelve fast carriers deposited by distance (criterion 1), the four
     slow ones by time (criterion 2);
  3. by distance: d_rec <= d < d_rec + |v| dt_max, i.e. within one step of
     reaching d_rec (|v| changes by g t = 1e-3 of itself, neglected);
  4. by time: t_rec <= t - t_launch < t_rec + dt_max;
  5. the logged d_rec equals max(0.5 kpc, 2 kernel_gamma h_IC) to 1e-6;
  6. the log's totals count four deposits by time.
"""

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check

from . import _common

DESCRIPTION = "wind carriers deposit within one step of d_rec or t_rec; criterion recorded"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.CARRIER_FLAGS

N = 32
L = 4.0  # kpc
NH = 0.01
T_GAS = 1e5
G_CGS = 1e-9
N_FAST, N_SLOW = 12, 4
V_FAST, V_SLOW = 300.0, 40.0
T_C = 3e6
D_REC = 0.5  # kpc
T_REC = 3.0  # Myr
T_END = 4.0  # Myr
DT_MAX = 0.01  # Myr
H0 = 1.2348 * L / N


def write_ics(path):
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pos)
    k = N_FAST + N_SLOW
    m = units.rho_from_nH(NH) * (L * units.KPC) ** 3 / n / units.UNIT_MASS
    rng = np.random.default_rng(7)
    cpos = 0.5 * L + rng.uniform(-0.5, 0.5, (k, 3))
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    speed = np.concatenate([np.full(N_FAST, V_FAST), np.full(N_SLOW, V_SLOW)])
    u = units.u_from_T(T_GAS, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    uc = units.u_from_T(T_C, units.MU_IONISED) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": pos,
        "Velocities": np.zeros((n, 3)),
        "Masses": np.full(n, m),
        "SmoothingLength": np.full(n, H0),
        "InternalEnergy": np.full(n, u),
        "ParticleIDs": np.arange(1, n + 1, dtype=np.uint64),
    }
    stars = {
        "Coordinates": cpos,
        "Velocities": d * speed[:, None],
        "Masses": np.full(k, m / 8.0),
        "SmoothingLength": np.full(k, H0),
        "ParticleIDs": np.arange(n + 1, n + k + 1, dtype=np.uint64),
        "WindCarrier": np.ones(k, np.int8),
        "WindCarrierInternalEnergies": np.full(k, uc),
    }
    ics.write(path, L, gas=gas, stars=stars)


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    write_ics(ctx.path("ics.hdf5"))
    myr = _common.MYR
    p = _common.base_params(ctx, "ics.hdf5", T_END * myr, 1.0 * myr, DT_MAX * myr,
                            g_cgs=(0.0, 0.0, G_CGS))
    p["Wind"] = _common.carrier_params(D_REC, T_REC)
    ctx.swift(p, FLAGS, name="carriers", timeout=1800)


def analyze(ctx):
    checks = []
    myr = _common.MYR
    n = N ** 3
    ev = sorted(_common.carrier_events(ctx.path("wind_carriers.txt")),
                key=lambda e: e["id"])
    tail = _common.carrier_tail(ctx.path("carriers.log"))
    checks.append(Check("deposits", len(ev), N_FAST + N_SLOW, kind="equal",
                        source="all reach d_rec or t_rec within the run"))
    checks.append(Check("log: retries (no gas in the kernel)",
                        tail["n_retry"] if tail else -1, 0, kind="equal",
                        source="uniform gas everywhere"))
    if len(ev) != N_FAST + N_SLOW:
        return checks
    idx = np.array([e["id"] - n - 1 for e in ev])
    fast = idx < N_FAST
    crit = np.array([e["criterion"] for e in ev])
    d = np.array([e["distance"] for e in ev])
    d_rec = np.array([e["d_rec"] for e in ev])
    age = np.array([e["t_deposit"] - e["t_launch"] for e in ev]) / myr
    speed = np.array([np.linalg.norm(e["v"]) for e in ev])
    dt = DT_MAX * myr

    checks.append(Check("fast carriers deposited by distance", int((crit[fast] == 1).sum()),
                        N_FAST, kind="equal", source="d_rec / v = 1.8 Myr < t_rec"))
    checks.append(Check("slow carriers deposited by time", int((crit[~fast] == 2).sum()),
                        N_SLOW, kind="equal", source="v t_rec = 0.12 kpc < d_rec"))
    checks.append(Check("by distance: min (d - d_rec) [kpc]", (d - d_rec)[fast].min(),
                        0.0, kind="min", source="not before d_rec"))
    checks.append(Check("by distance: max (d - d_rec) / (|v| dt_max)",
                        ((d - d_rec) / (speed * dt))[fast].max(), 1.0, kind="max",
                        source="within one step"))
    checks.append(Check("by time: min (t - t_launch - t_rec) [Myr]",
                        (age - T_REC)[~fast].min(), -1e-9, kind="min",
                        source="not before t_rec"))
    checks.append(Check("by time: max (t - t_launch - t_rec) [Myr]",
                        (age - T_REC)[~fast].max(), DT_MAX * (1 + 1e-6), kind="max",
                        source="within one dt_max"))
    expect = max(D_REC, 2 * _common.KERNEL_GAMMA * H0)
    checks.append(Check("logged d_rec", d_rec.min(), expect, kind="rel", tol=1e-6,
                        source="max(0.5 kpc, 2 kernel_gamma h_IC)"))
    checks.append(Check("log: deposits by time", tail["n_by_time"] if tail else -1,
                        N_SLOW, kind="equal", source="Wind carrier totals line"))
    ctx.info.update({"age_Myr": age.tolist(), "d_kpc": d.tolist(),
                     "criterion": crit.tolist()})
    return checks
