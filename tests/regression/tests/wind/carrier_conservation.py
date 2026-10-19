"""Wind carriers deposit their mass, momentum and energy conservatively (X1).

A periodic 4 kpc box of gas on a 32^3 lattice (n_H = 0.01 cm^-3, T = 1e5 K,
no cooling, no star formation: n_H is below the PRFM threshold) in a
constant field g = 1e-9 cm s^-2 along z. Two star particles are flagged
WindCarrier = 1 in the ICs, i.e. carriers launched at t = 0, each of mass
m_g / 4 with T_c = 3e6 K (mu = 0.59): A moves at 300 km/s along +x from
(1, 2, 2) kpc and deposits by distance (d_rec = max(0.5 kpc, 2 kernel
supports) = 0.56 kpc, reached after 1.8 Myr); B moves at 40 km/s along -x
from (3, 2, 2) kpc and deposits by time (t_rec = 3 Myr, after 0.12 kpc).
Wind:mode = carrier, run with --stars --feedback (the carriers use the stars
density and feedback loops). dt_max = 0.01 Myr, snapshots every 0.1 Myr,
4 Myr.

Every particle, carriers included, feels the same g, so the lattice falls
rigidly until the first deposit, and a uniform field makes the potential
energy of a moved mass exactly dm a.(x_c - x_j), the term the deposit takes
from the thermal energy.

Checks:
  1. both carriers deposited and were removed (no star particle at the end,
     two lines in wind_carriers.txt), A by distance, B by time;
  2. total mass (gas + carriers) at every snapshot against t = 0, relative
     1e-6 (masses stored as float, 32770 summands);
  3. momentum. Along z, P_z - P_z(0) - M g t within M g dt_max / 2: the
     output velocity is v_full + a dt/2 (recouple.py), dt <= dt_max. In x
     and y the lattice drifts steadily from t = 0, before any carrier
     deposits, by 6e-5 m_c |v_A| per Myr (float rounding of the SPH forces;
     2e-4 over the run). The check is that a 0.1 Myr interval holding a
     deposit changes P_xy by no more than 3 times the largest change of an
     interval before the first deposit. The deposit itself conserves
     momentum to round-off (tests/testWindCarrier.c);
  4. total energy sum m (v^2/2 + u - g z) of gas and carriers (u_c for a
     carrier) at every snapshot against t = 0, relative 1e-5: the bound of
     recouple.py, where hydro time integration of the shocks the deposit
     drives is the only non-exact step. The carriers hold 1.5 per cent of E;
  5. each gas particle's mass gain against its kernel share
     w_j = W(|x_j - x_c|, h_c) / sum W of the carrier's mass, with x_c and
     h_c from wind_carriers.txt and x_j the lattice at the deposit time
     (fallen by g t^2 / 2): max |dm_j / m_c - w_j| < 1e-5 (float masses,
     dm_j / m_g ~ 1e-2); every particle with a share above 1e-5 gained
     mass (smaller gains are below the float resolution of m_g), none
     outside the kernels; the lattice count inside the kernels equals the
     logged neighbour count, and sum W the logged weight sum to 1e-5;
  6. the log's carrier totals: deposited mass = launched mass (1e-6), no
     energy added by the internal-energy floor.
"""

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common

DESCRIPTION = "wind carriers: deposit conserves mass, momentum and energy (with potential); shares = kernel weights"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = _common.CARRIER_FLAGS

N = 32
L = 4.0  # kpc
NH = 0.01
T_GAS = 1e5
G_CGS = 1e-9
T_C = 3e6
MASS_FRACTION = 0.25
XA, VA = (1.0, 2.0, 2.0), (300.0, 0.0, 0.0)
XB, VB = (3.0, 2.0, 2.0), (-40.0, 0.0, 0.0)
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
    m = units.rho_from_nH(NH) * (L * units.KPC) ** 3 / n / units.UNIT_MASS
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
        "Coordinates": np.array([XA, XB]),
        "Velocities": np.array([VA, VB]),
        "Masses": np.full(2, MASS_FRACTION * m),
        "SmoothingLength": np.full(2, H0),
        "ParticleIDs": np.array([n + 1, n + 2], dtype=np.uint64),
        "WindCarrier": np.ones(2, np.int8),
        "WindCarrierInternalEnergies": np.full(2, uc),
    }
    ics.write(path, L, gas=gas, stars=stars)


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    write_ics(ctx.path("ics.hdf5"))
    myr = _common.MYR
    p = _common.base_params(ctx, "ics.hdf5", T_END * myr, 0.1 * myr, DT_MAX * myr,
                            g_cgs=(0.0, 0.0, G_CGS))
    p["Wind"] = _common.carrier_params(D_REC, T_REC)
    ctx.swift(p, FLAGS, name="carriers", timeout=1800)


def totals(snap, g):
    """Mass, momentum and energy of gas and carriers (internal units)."""
    m = snap.read(0, "Masses", cgs=False)
    v = snap.read(0, "Velocities", cgs=False)
    u = snap.read(0, "InternalEnergies", cgs=False)
    z = snap.read(0, "Coordinates", cgs=False)[:, 2]
    M = m.sum()
    P = (m[:, None] * v).sum(axis=0)
    E = np.sum(m * (0.5 * (v ** 2).sum(axis=1) + u - g * (z - 0.5 * L)))
    if snap.npart[4] > 0:
        ms = snap.read(4, "Masses", cgs=False)
        vs = snap.read(4, "Velocities", cgs=False)
        us = snap.read(4, "WindCarrierInternalEnergies", cgs=False)
        zs = snap.read(4, "Coordinates", cgs=False)[:, 2]
        M += ms.sum()
        P += (ms[:, None] * vs).sum(axis=0)
        E += np.sum(ms * (0.5 * (vs ** 2).sum(axis=1) + us - g * (zs - 0.5 * L)))
    return M, P, E


def kernel(q):
    """Cubic spline of x = r / (gamma h), unnormalised (shares only)."""
    x = q / _common.KERNEL_GAMMA
    return np.where(x < 0.5, 1 - 6 * x ** 2 + 6 * x ** 3,
                    np.where(x < 1, 2 * (1 - x) ** 3, 0.0))


def analyze(ctx):
    checks = []
    snaps = snapshots(ctx.run_dir)
    myr = _common.MYR
    g = G_CGS / _common.ACC_UNIT
    first, last = snaps[0], snaps[-1]
    ev = _common.carrier_events(ctx.path("wind_carriers.txt"))

    checks.append(Check("carriers left at the end", int(last.npart[4]), 0, kind="equal",
                        source="both deposit within the run and are removed"))
    checks.append(Check("deposit events", len(ev), 2, kind="equal",
                        source="wind_carriers.txt"))
    crit = {int(e["id"]) - N ** 3: int(e["criterion"]) for e in ev}
    checks.append(Check("A deposited by distance (1)", crit.get(1, -1), 1, kind="equal",
                        source="d_rec / |v_A| = 1.8 Myr < t_rec"))
    checks.append(Check("B deposited by time (2)", crit.get(2, -1), 2, kind="equal",
                        source="|v_B| t_rec = 0.12 kpc < d_rec"))

    M0, P0, E0 = totals(first, g)
    m_c = first.read(4, "Masses", cgs=False)
    p_c = m_c[0] * np.linalg.norm(VA)
    dM, Pxy, dPz, dE, times = [], [], [], [], []
    for s in snaps:
        M, P, E = totals(s, g)
        t = s.time_cgs / units.UNIT_TIME
        times.append(t)
        dM.append(abs(M / M0 - 1))
        Pxy.append((P - P0)[:2] / p_c)
        dPz.append(abs(P[2] - P0[2] - M0 * g * t) / (M0 * g * DT_MAX * myr))
        dE.append(abs(E / E0 - 1))
    checks.append(Check("max |M(t) / M(0) - 1|", max(dM), 1e-6, kind="max",
                        source="gas + carriers, float masses"))
    # P_xy drifts steadily from t = 0, before any deposit (float rounding of
    # the lattice's SPH forces); a deposit must not add to the drift
    times = np.array(times)
    inc = np.linalg.norm(np.diff(np.array(Pxy), axis=0), axis=1)
    t_first = min(e["t_deposit"] for e in ev) if ev else times[-1]
    before = times[1:] < t_first
    bracket = np.zeros(len(inc), bool)
    for e in ev:
        bracket |= (times[:-1] < e["t_deposit"]) & (times[1:] >= e["t_deposit"])
    bg = inc[before].max() if before.any() else 0.0
    checks.append(Check("max |dP_xy| over an interval holding a deposit / m_c |v_A|",
                        inc[bracket].max() if bracket.any() else -1, 3 * bg,
                        kind="max", source=f"3 x the largest change per 0.1 Myr before "
                                           f"the first deposit ({bg:.2e})"))
    checks.append(Check("max |P_z(t) - P_z(0) - M g t| / (M g dt_max)", max(dPz), 0.5,
                        kind="max", source="output velocity v_full + g dt/2 "
                                           "(recouple.py), dt <= dt_max"))
    checks.append(Check("max |E(t) / E(0) - 1|", max(dE), 1e-5, kind="max",
                        source="sum m (v^2/2 + u - g z), carriers with u_c"))

    # Kernel shares, from the gas mass gains
    n = N ** 3
    ids = last.read(0, "ParticleIDs", cgs=False).astype(np.int64)
    o = np.argsort(ids)
    m_end = last.read(0, "Masses", cgs=False)[o]
    m_ini = first.read(0, "Masses", cgs=False)[np.argsort(
        first.read(0, "ParticleIDs", cgs=False).astype(np.int64))]
    x = (np.arange(N) + 0.5) * (L / N)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    lat = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    gained = m_end - m_ini
    w_all = np.zeros(n)
    n_recv, n_log, sum_err = 0, 0, 0.0
    for e in ev:
        t_d = e["t_deposit"]
        xc = np.array(e["x_deposit"])
        pos = lat.copy()
        pos[:, 2] += 0.5 * g * t_d ** 2
        d = pos - xc
        d -= L * np.round(d / L)
        q = np.linalg.norm(d, axis=1) / e["h"]
        w = kernel(q)
        sum_err = max(sum_err, abs(w.sum() * _norm_w() / e["weight_sum"] - 1))
        w_all += w / w.sum() * e["mass"]
        n_recv += int((w > 0).sum())
        n_log += int(e["n_ngb"])
    max_err = np.abs(gained - w_all).max() / (MASS_FRACTION * m_ini[0])
    checks.append(Check("max |dm_j - w_j m_c| / m_c", max_err, 1e-5, kind="max",
                        source="kernel shares at the logged x_c, h_c"))
    big = w_all > 1e-5 * MASS_FRACTION * m_ini[0]
    checks.append(Check("receivers with share > 1e-5 that gained mass",
                        int((gained[big] > 0).sum()), int(big.sum()), kind="equal",
                        source="smaller shares are below float resolution of m_g"))
    checks.append(Check("gas that gained mass outside the kernels",
                        int(((gained > 0) & (w_all == 0)).sum()), 0, kind="equal",
                        source="deposit only inside kernel_gamma h_c"))
    checks.append(Check("receivers vs logged neighbour counts", n_recv, n_log,
                        kind="equal", source="wind_carriers.txt"))
    checks.append(Check("max |sum W / logged weight sum - 1|", sum_err, 1e-5, kind="max",
                        source="kernel sum recomputed from the lattice"))

    tail = _common.carrier_tail(ctx.path("carriers.log"))
    checks.append(Check("log: deposited mass / launched mass",
                        tail["m_deposit"] / tail["m_launch"] if tail else -1, 1.0,
                        kind="rel", tol=1e-6, source="Wind carrier totals line"))
    checks.append(Check("log: energy added by the u floor", tail["E_floor"] if tail else -1,
                        0.0, kind="abs", tol=0.0, source="no deposit hits the floor"))
    ctx.info.update({
        "events": ev, "dM_max": max(dM), "Pxy_end_over_pc": np.linalg.norm(Pxy[-1]), "dPz_max": max(dPz), "dE_max": max(dE),
        "log_tail": tail})
    return checks


def _norm_w():
    """kernel_eval(u) of kernel_hydro.h (cubic spline) over kernel() above:
    kernel_constant / gamma^3 = (16 / pi) / gamma^3 times SWIFT's polynomial,
    which is half of kernel(); the deposit sums kernel_eval without h^-3."""
    return 8.0 / np.pi / _common.KERNEL_GAMMA ** 3
