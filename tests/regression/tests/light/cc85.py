"""V3: the Chevalier & Clegg (1985) wind from a central source, built from light
particles (long tier).

A periodic box of 40 kpc holds a uniform ambient medium, n_H = 1e-5 cm^-3,
T = 1e6 K, at rest, of 14^3 particles (m_g = 7.6e3 Msun). A central source
injects mass and energy at Mdot = 0.285 Msun/yr and Edot = 5.5e40 erg/s
(light-particles-design.md Sec. 1.3, the TIGRESS hot loadings of a Milky-Way
disc) as light particles of m_l = m_g / 10 = 760 Msun (Wind:light_source_*),
created at tree rebuilds every 0.1 Myr at uniformly random points inside
R = 0.3 kpc and launched radially at Mach 1 with kinetic plus internal energy
e_w = Edot / Mdot per unit mass (Wind:energy_convention = internal). They fly
0.05 kpc decoupled and recouple. No gravity, no cooling. SPHENIX and
GIZMO-MFM; run to 20 Myr.

The internal-energy convention is used because it makes Edot the energy the
particles actually carry, which is the CC85 energy injection rate. With the
enthalpy convention a Mach-1 launch carries kinetic plus internal energy of
only 0.7 e_w (Delta v^2/2 = e_w / 4, u_w = 3 e_w / (4 gamma)), and a first
run (2026-10-20) measured an energy flux of 0.69-0.72 Edot and a Bernoulli
speed of 650 km/s against (2 x 0.7 e_w)^1/2 = 654 km/s: the flow conserves
the energy the particles carry, and the PdV work counted by the enthalpy is
not extra energy (design Sec. 6, note to stage 2).

Analytic solution (Chevalier & Clegg 1985, for r > R): with Mach number M,
   M^(2/(gamma-1)) [(gamma - 1 + 2/M^2) / (gamma + 1)]^((gamma+1)/(2(gamma-1)))
     = (r / R)^2,
v = M (2 e_w / (M^2 + 2/(gamma-1)))^1/2, rho = Mdot / (4 pi r^2 v),
P = rho c^2 / gamma; v_inf = (2 e_w)^1/2 = 782 km/s.

Measured at the last snapshot between 1 and 5 kpc, in 12 logarithmic radial
bins (gas particles that are not decoupled): mass-weighted v_r; the density
and pressure as the volume-weighted bin averages sum m / sum (m / rho) and
sum (m P / rho) / sum (m / rho); through shells of width 0.2 r at 2 and
4 kpc the mass flux sum(m v_r) / dr and the energy flux
sum(m v_r (v^2/2 + gamma u)) / dr; and the mass-weighted Bernoulli speed
(v^2 + 2 gamma u)^1/2 in those shells. The injection radius of the particle
run is not sharp (random points inside R, a kernel of 0.1-0.3 kpc at the
source), so the effective R of the measured profile is recorded by fitting
CC85 to the velocity profile over R in [0.3, 2] kpc.

Checks, per scheme:
  1. mass injected (log) against Mdot t, to Mdot times one rebuild interval
     plus 2 m_l (each rebuild spawns the floor of the mass due over m_l and
     carries the rest; the last rebuild may precede the snapshot);
  2. the free wind reaches 5 kpc: v_r in the outermost bin above
     0.8 v_CC (otherwise the termination shock is inside the range);
  3. mass flux at 2 and 4 kpc against Mdot within 15 per cent and energy
     flux against Edot within 15 per cent (about 200 particles per shell,
     arranged by the flow rather than at random);
  4. Bernoulli speed at 2 and 4 kpc against v_inf within 5 per cent: the
     total energy per mass is conserved along the flow;
  5. median |v_r / v_CC - 1| over the bins below 0.1 (design V3: speed
     within 10 per cent);
  6. median |rho / rho_CC - 1| below 0.3 and median |ln(P / P_CC)| below
     ln 4: the pressure of a flow at Mach 3-10 is a small remainder of its
     energy, so a shift of the effective injection radius or a little
     entropy made by the scheme changes it by a larger factor than the
     density. Measured 2026-10-20 at 20 Myr: P / P_CC = 1.5 at every radius
     (SPHENIX) and 2.3 at 1 kpc rising to 3.7 at 5 kpc (MFM), with
     v_r / v_CC = 0.96-0.99 and rho / rho_CC = 1.05-1.17 in both.
"""

import os

import numpy as np
from scipy.optimize import brentq

from swiftreg import ics, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..wind import _common as wind_common
from . import _common

DESCRIPTION = "V3: CC85 wind from light particles against the analytic profiles"
TIER = "long"
BUILDS = {"sph": _common.SPH_BUILD, "mfm": _common.MFM_BUILD}

BOX = 40.0  # kpc
N_SIDE = 14
NH, T_AMB = 1e-5, 1e6
MDOT = 0.285  # Msun / yr
EDOT = 5.5e40  # erg / s
R_SRC = 0.3  # kpc
D_REC = 0.05  # kpc
N_RATIO = 10
T_END_MYR = 20.0
DT_REBUILD_MYR = 0.1
GAMMA = 5.0 / 3.0
R_LO, R_HI, N_BINS = 1.0, 5.0, 12
MYR = units.MYR / units.UNIT_TIME


def e_w_cgs():
    return EDOT / (MDOT * units.SOLAR_MASS / units.YEAR)


def cc85(r_kpc, R=R_SRC):
    """(rho [g/cm^3], v [cm/s], P [erg/cm^3]) of CC85 outside R [kpc]."""
    e_w = e_w_cgs()
    mdot = MDOT * units.SOLAR_MASS / units.YEAR
    g = GAMMA
    out = []
    for r in np.atleast_1d(r_kpc):
        x2 = (r / R) ** 2

        def f(M):
            return (2 / (g - 1)) * np.log(M) + (g + 1) / (2 * (g - 1)) * np.log(
                (g - 1 + 2 / M ** 2) / (g + 1)) - np.log(x2)

        M = brentq(f, 1.0 + 1e-12, 1e4) if r > R else 1.0
        v = M * np.sqrt(2 * e_w / (M ** 2 + 2 / (g - 1)))
        rho = mdot / (4 * np.pi * (r * units.KPC) ** 2 * v)
        c2 = v ** 2 / M ** 2
        out.append((rho, v, rho * c2 / g))
    return np.array(out)


def run(ctx):
    ctx.table(wind_common.params.PRFM_COOLING_TABLE)
    n = N_SIDE
    gas, m_g_Msun = ics.uniform_gas(n, BOX, NH, T_AMB, mu=units.MU_IONISED)
    # A lattice of N_SIDE^3 fixes m_g; the box is filled at the requested density
    ctx.info["m_g_Msun"] = m_g_Msun
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas)
    for b in BUILDS:
        sub = b
        os.makedirs(ctx.path(sub), exist_ok=True)
        if not os.path.exists(ctx.path(sub, "ics.hdf5")):
            os.symlink(ctx.path("ics.hdf5"), ctx.path(sub, "ics.hdf5"))
        p = wind_common.base_params(ctx, "ics.hdf5", T_END_MYR * MYR, 2.0 * MYR,
                                    0.05 * MYR)
        # Four top-level cells of 10 kpc; ambient particles need h of about
        # 3.5 kpc, and kernel_gamma h_max 1.1 must stay below the cell
        p["Scheduler"]["max_top_level_cells"] = 4
        p["SPH"]["h_max"] = 4.5
        p["SPH"]["minimal_temperature"] = 100.0
        p["Wind"] = {"loading_model": "none", "energy_convention": "internal",
                     "recoupling_distance_kpc": D_REC, "recoupling_kernel_factor": 0.0,
                     "recoupling_time_Myr": 1e4, "max_timestep_Myr": 0.05,
                     **_common.light_params(m_g_Msun, N_RATIO,
                                            light_rebuild_interval_Myr=DT_REBUILD_MYR,
                                            light_R200_kpc=200.0,
                                            light_source_Mdot_Msun_p_yr=MDOT,
                                            light_source_Edot_erg_p_s=EDOT,
                                            light_source_radius_kpc=R_SRC,
                                            light_source_centre=[0.5 * BOX] * 3)}
        flags = list(wind_common.NOSF_FLAGS)
        if b == "mfm":
            flags += ["--limiter", "--sync"]
        ctx.swift(p, flags, binary=b, subdir=sub, name="run", timeout=6 * 3600)


def profiles(snap):
    x = snap.read(0, "Coordinates") / units.KPC - 0.5 * BOX
    v = snap.read(0, "Velocities")
    m = snap.read(0, "Masses")
    rho = snap.read(0, "Densities")
    P = snap.read(0, "Pressures")
    u = snap.read(0, "InternalEnergies")
    keep = snap.read(0, "WindDecoupled", cgs=False) == 0
    x, v, m, rho, P, u = x[keep], v[keep], m[keep], rho[keep], P[keep], u[keep]
    r = np.linalg.norm(x, axis=1)
    vr = np.sum(x * v, axis=1) / np.maximum(r, 1e-30)
    edges = np.geomspace(R_LO, R_HI, N_BINS + 1)
    rc, prof = np.sqrt(edges[1:] * edges[:-1]), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (r >= lo) & (r < hi)
        if s.sum() == 0:
            prof.append((np.nan, np.nan, np.nan, 0))
            continue
        vol = (m[s] / rho[s]).sum()
        prof.append(((m[s] * vr[s]).sum() / m[s].sum(), m[s].sum() / vol,
                     (m[s] * P[s] / rho[s]).sum() / vol, int(s.sum())))
    prof = np.array(prof)
    fluxes = {}
    for r0 in (2.0, 4.0):
        dr = 0.2 * r0
        s = np.abs(r - r0) < 0.5 * dr
        e = 0.5 * np.sum(v[s] ** 2, axis=1) + GAMMA * u[s]
        fluxes[r0] = {
            "Mdot": float((m[s] * vr[s]).sum() / (dr * units.KPC)),
            "Edot": float((m[s] * vr[s] * e).sum() / (dr * units.KPC)),
            "v_B": float(np.sqrt(np.average(2 * e, weights=m[s]))),
            "n": int(s.sum())}
    return rc, prof, fluxes


def fit_R(rc, v):
    """CC85 injection radius [kpc] that best fits a velocity profile."""
    grid = np.linspace(0.3, 2.0, 69)
    err = [np.mean(np.log(v / cc85(rc, R)[:, 1]) ** 2) for R in grid]
    return float(grid[int(np.argmin(err))])


def analyze(ctx):
    checks = []
    mdot = MDOT * units.SOLAR_MASS / units.YEAR
    v_inf = np.sqrt(2 * e_w_cgs())
    for b in BUILDS:
        snap = snapshots(ctx.path(b))[-1]
        rc, prof, fluxes = profiles(snap)
        ex = cc85(rc)
        good = prof[:, 3] > 0
        dv = np.abs(prof[good, 0] / ex[good, 1] - 1)
        drho = np.abs(prof[good, 1] / ex[good, 0] - 1)
        dlogP = np.abs(np.log(prof[good, 2] / ex[good, 2]))
        tail = _common.light_tail(ctx.path(b, "run.log"))
        m_src = tail["m_source"] * units.UNIT_MASS / units.SOLAR_MASS if tail else 0.0
        t_myr = snap.time_cgs / units.MYR
        m_l = ctx.info["m_g_Msun"] / N_RATIO
        checks.append(Check(f"{b}: injected mass [Msun]", m_src,
                            MDOT * 1e6 * t_myr, kind="abs",
                            tol=MDOT * 1e6 * DT_REBUILD_MYR + 2 * m_l,
                            source="floor of the mass due over m_l per rebuild, the "
                                   "rest carried; the last rebuild is up to one "
                                   "interval before the snapshot"))
        checks.append(Check(f"{b}: v_r / v_CC in the outermost bin", prof[-1, 0] / ex[-1, 1],
                            0.8, kind="min", source="free wind beyond 5 kpc"))
        for r0, f in fluxes.items():
            checks.append(Check(f"{b}: Mdot at {r0:g} kpc / Mdot", f["Mdot"] / mdot, 1.0,
                                kind="rel", tol=0.15, source="mass conservation"))
            checks.append(Check(f"{b}: Edot at {r0:g} kpc / Edot", f["Edot"] / EDOT, 1.0,
                                kind="rel", tol=0.15,
                                source="energy conservation (kinetic + enthalpy flux)"))
            checks.append(Check(f"{b}: v_B at {r0:g} kpc / v_inf", f["v_B"] / v_inf, 1.0,
                                kind="rel", tol=0.05,
                                source="Bernoulli: v^2/2 + gamma u = e_w"))
        checks.append(Check(f"{b}: median |v_r / v_CC - 1|", np.median(dv), 0.1, kind="max",
                            source="design V3: speed within 10%"))
        checks.append(Check(f"{b}: median |rho / rho_CC - 1|", np.median(drho), 0.3,
                            kind="max", source="kernel average of an r^-2 profile"))
        checks.append(Check(f"{b}: median |ln(P / P_CC)|", np.median(dlogP), np.log(4.0),
                            kind="max", source="pressure is a small remainder at Mach 3-10; "
                                               "measured 1.5 (SPHENIX), 2.3-3.7 (MFM)"))
        R_fit = fit_R(rc[good], prof[good, 0])
        ctx.info[b] = {"t_Myr": t_myr, "r": rc.tolist(), "v": prof[:, 0].tolist(),
                       "rho": prof[:, 1].tolist(), "P": prof[:, 2].tolist(),
                       "n": prof[:, 3].tolist(), "R_fit_kpc": R_fit,
                       "fluxes": {str(k): v for k, v in fluxes.items()},
                       "m_source_Msun": m_src,
                       "median_dv": float(np.median(dv)),
                       "median_drho": float(np.median(drho)),
                       "median_dlogP": float(np.median(dlogP))}
    return checks
