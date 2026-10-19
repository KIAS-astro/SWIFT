"""Wind launch from PRFM star formation in a uniform gas layer.

The star-forming layer of star_formation/prfm_slab (1 x 1 x 0.5 kpc periodic
box, Sigma_g = 200 Msun/pc^2, every particle star-forming at
mdot_* = m_g / t_dep) with the decoupled wind on at constant loadings
eta_M = 0.5, eta_E = 0.1 (Wind:loading_model = constant, whose default
Wind:energy_convention = internal counts kinetic + internal energy in eta_E).
Each step one uniform number decides star (u < p_*), wind
(p_* <= u < p_* + p_w, p_w = eta_M p_*) or nothing (wind-design.md WD1).
Recoupling is pushed beyond
the run (d_rec = 1e3 kpc, t_rec = 1e4 Myr), so every launched particle is
still decoupled at the end and, with no gravity (constant zero field) and
no hydro force, still moves with its launch velocity.

Run for 2 per cent of t_dep (about 640 stars from 32000 particles). Checks:
  1. launches N_w against eta_M N_*: the two counts are draws of one
     multinomial, so N_w - eta_M N_* has variance eta_M (1 + eta_M) E[N_*];
     within 3 sigma;
  2. internal energy of every launched particle at launch against
     u_w = eta_E,th u_* / eta_M (WD3), u_* = 1e51 erg / 95.5 Msun,
     eta_E,th = 9/14 eta_E, to 1e-6 (float storage);
  3. mean launch speed sqrt(2 e_kin) against
     Delta v = (2 eta_E,kin u_* / eta_M)^1/2, eta_E,kin = 5/14 eta_E, to
     1e-3 (gas speeds before launch, a few km/s from the holes left by
     converted particles, average out over the random signs);
  4. every launched particle's velocity differs from +-Delta v z by no more
     than the fastest coupled gas particle (its velocity before the kick),
     and the fraction moving up is 1/2 within 3 binomial sigma;
  5. fluxes: launched mass and launched energy (kinetic + internal) against
     eta_M M_* and eta_E u_* M_* within 3 sigma, and launched energy per mass
     against eta_E u_* / eta_M to 1e-3;
  6. the log's wind totals count the same launches as the snapshot.
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _common

DESCRIPTION = "wind launch: launched mass / stars formed = eta_M; launch speed and u_w as WD3"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

SIGMA = 200.0  # Msun / pc^2
ETA_M, ETA_E = 0.5, 0.1
F_KIN = 5.0 / 14.0
FRACTION = 0.02
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g


def run(ctx):
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    Sigma = slab.write_layer(ctx.path("ics.hdf5"), SIGMA)
    t_dep = slab.t_dep_layer(Sigma) / units.UNIT_TIME
    t_end = FRACTION * t_dep
    p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, t_end,
                          dt_max=slab.DT_MAX * t_dep, cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
    p["SPH"]["h_max"] = 2.0 * 1.2348 * slab.LX / slab.NX
    p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
    p["PRFMStarFormation"] = {
        "calibration": "OK22", "threshold": "measured",
        "threshold_n_H_p_cm3": 0.13, "use_dark_matter": 0,
        "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
        "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
        "map_half_width_kpc": 0.5 * slab.LX}
    p["Wind"] = {"loading_model": "constant", "eta_M": ETA_M, "eta_E": ETA_E,
                 "launch_direction": [0.0, 0.0, 1.0],
                 "recoupling_distance_kpc": 1e3, "recoupling_time_Myr": 1e4}
    ctx.info.update({"t_dep_Myr": t_dep * units.UNIT_TIME / units.MYR,
                     "t_end_Myr": t_end * units.UNIT_TIME / units.MYR})
    ctx.swift(p, FLAGS, name="wind", timeout=1800)


def analyze(ctx):
    checks = []
    last = snapshots(ctx.run_dir)[-1]
    n_star = int(last.npart[4])
    counts = last.read(0, "WindLaunchCounts", cgs=False)
    n_wind = int(counts.sum())
    launched = counts > 0
    dec = last.read(0, "WindDecoupled", cgs=False) > 0
    sigma = np.sqrt(ETA_M * (1.0 + ETA_M) * max(n_star, 1))
    checks.append(Check("launches / eta_M stars", n_wind, ETA_M * n_star,
                        kind="sigma", tol=3.0, sigma=sigma,
                        source=f"one uniform number per step, p_w = eta_M p_*; "
                               f"N_* = {n_star}; sigma^2 = eta_M (1 + eta_M) N_*"))
    checks.append(Check("launched particles still decoupled", (dec == launched).all(), 1,
                        kind="equal", source="recoupling beyond the run"))

    e_w = ETA_E * U_STAR / ETA_M
    u_w = (1.0 - F_KIN) * e_w
    dv = np.sqrt(2.0 * F_KIN * e_w)
    u_launch = last.read(0, "WindLaunchInternalEnergies")[launched]
    e_kin = last.read(0, "WindLaunchKineticEnergies")[launched]
    checks.append(Check("max |u_launch / u_w - 1|", np.abs(u_launch / u_w - 1).max(),
                        1e-6, kind="max",
                        source=f"u_w = 9/14 eta_E u_* / eta_M = "
                               f"{units.T_from_u(u_w, 0.6):.3g} K (mu = 0.6)"))
    checks.append(Check("mean launch speed [km/s]", np.sqrt(2 * e_kin).mean() / 1e5,
                        dv / 1e5, kind="rel", tol=1e-3,
                        source="Delta v = (2 5/14 eta_E u_* / eta_M)^1/2"))
    v = last.read(0, "Velocities")[launched]
    v_gas = np.linalg.norm(last.read(0, "Velocities")[~dec], axis=1).max()
    kick = np.zeros_like(v)
    kick[:, 2] = np.sign(v[:, 2]) * dv
    checks.append(Check("max |v - (+-Delta v) z| [km/s]",
                        np.linalg.norm(v - kick, axis=1).max() / 1e5, v_gas / 1e5,
                        kind="max", source="kick along +-z added to a gas velocity no "
                                           "larger than the fastest coupled gas "
                                           "particle at the end; ballistic"))
    n_up = int((v[:, 2] > 0).sum())
    checks.append(Check("launches upwards", n_up, 0.5 * n_wind, kind="sigma", tol=3.0,
                        sigma=0.5 * np.sqrt(max(n_wind, 1)),
                        source="random sign, binomial(N_w, 1/2)"))
    # Fluxes: launched mass and energy against eta_M and eta_E times the
    # stellar mass formed and its SN energy (the same Poisson sigma, scaled)
    m_g = last.read(0, "Masses")[launched]
    m_star = last.read(4, "InitialMasses").sum()
    E_launch = np.sum(m_g * (u_launch + e_kin))
    m_w = m_g.sum()
    checks.append(Check("launched mass / eta_M M_*", m_w / (ETA_M * m_star), 1.0,
                        kind="sigma", tol=3.0, sigma=sigma / (ETA_M * max(n_star, 1)),
                        source="Mdot_w = eta_M SFR"))
    checks.append(Check("launched energy / eta_E u_* M_*", E_launch / (ETA_E * U_STAR * m_star),
                        1.0, kind="sigma", tol=3.0, sigma=sigma / (ETA_M * max(n_star, 1)),
                        source="Edot_w = eta_E u_* SFR; kinetic + internal at launch"))
    checks.append(Check("launched energy per mass / (eta_E u_* / eta_M)",
                        E_launch / m_w / e_w, 1.0, kind="rel", tol=1e-3,
                        source="WD3; the gas kinetic energy before launch adds "
                               "< 1e-3"))
    tail = _common.wind_tail(ctx.path("wind.log"))
    checks.append(Check("log: launches", tail["n_launch"] if tail else -1, n_wind,
                        kind="equal", source="Wind totals line at the last statistics"))
    ctx.info.update({"stars": n_star, "launches": n_wind, "dv_kms": dv / 1e5,
                     "T_w_K": units.T_from_u(u_w, 0.6),
                     "log_tail": tail, "m_launched_Msun": m_g.sum() / units.SOLAR_MASS})
    return checks
