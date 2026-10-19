"""Light wind particles spawned from PRFM star formation in a uniform layer.

The star-forming layer of star_formation/prfm_slab (1 x 1 x 0.5 kpc periodic
box, Sigma_g = 200 Msun/pc^2, 32000 particles of m_g = 6250 Msun, every
particle star-forming at mdot_* = m_g / t_dep) with Wind:mode light,
constant loadings eta_M = 0.5, eta_E = 0.1 (internal-energy convention),
N = 10 (m_l = 625 Msun) and a forced rebuild every t_end / 10. Each
star-forming particle adds eta_M mdot_* dt to its reservoir every step; at
each rebuild every reservoir spawns m_pending / m_l light particles rounded
at random to one of the two nearest integers (expected value kept) and is
emptied, the remainder ("rounding", zero in expectation) staying in the
parent. They are launched along +-z and stay decoupled to the end
(d_rec = 1e3 kpc, t_rec = 1e4 Myr). Run for 2 per cent of t_dep.

Checks:
  1. reservoir bookkeeping, from the log: mass added to reservoirs = mass
     spawned + rounding + mass held in reservoirs + mass that left with
     stars, to 1e-9 (double sums of float masses, 10-digit log);
  2. mass added to reservoirs against eta_M times the integrated SFR
     sum(SFR dt) of SFR.txt, to 1e-4 (7-digit columns summed over ~2000
     steps);
  3. rounding / added within 3 sigma of zero: a draw of mean x m_l has
     variance x (1 - x) m_l^2 <= x m_l^2, so the sum has variance at most
     m_spawned m_l;
  4. spawned + held + left with stars against eta_M times the stellar mass
     formed, within 3 sigma: the stars are a Poisson draw
     (eta_M m_g N_*^1/2) and the rounding adds (m_spawned m_l)^1/2;
  5. mass that left with stars: a reservoir holds at most the wind of one
     rebuild interval and a particle forms a star once per t_dep, so
     lost / added stays below 3 dt_rebuild / t_dep;
  6. snapshot: the number and mass of WindLight particles equal the log's
     spawned count and mass (exact count; mass to 1e-6, float masses), every
     light particle has mass m_l (1e-6) and is decoupled;
  7. launch internal energy of every light particle against
     u_w = (1 - f) eta_E u_* / eta_M, f = 5/14, to 1e-6, and the launch speed
     relative to the parent, |v_z - v_z,parent| = Delta v, through the mean
     of |v_z| to 1e-2 (parents move at a few km/s);
  8. global mass: gas + stars at the end against the initial gas mass, to
     1e-6 (float masses).
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _common

DESCRIPTION = "light wind: reservoir bookkeeping exact; created mass = eta_M x SFR"
TIER = "quick"
BUILDS = {"sph": _common.SPH_BUILD}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

SIGMA = 200.0  # Msun / pc^2
ETA_M, ETA_E = 0.5, 0.1
F_KIN = 5.0 / 14.0
N_LIGHT = 10
FRACTION = 0.02
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g


def run(ctx):
    ctx.swift(layer_params(ctx, FRACTION), FLAGS, name="light", timeout=1800)


def layer_params(ctx, fraction):
    """ICs and parameters of the light-wind layer run for fraction of t_dep."""
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    Sigma = slab.write_layer(ctx.path("ics.hdf5"), SIGMA)
    t_dep = slab.t_dep_layer(Sigma) / units.UNIT_TIME
    t_end = fraction * t_dep
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
    m_g = _gas_mass_Msun(ctx.path("ics.hdf5"))
    dt_rebuild_Myr = 0.1 * t_end * units.UNIT_TIME / units.MYR
    p["Wind"] = {"loading_model": "constant", "eta_M": ETA_M, "eta_E": ETA_E,
                 "launch_direction": [0.0, 0.0, 1.0],
                 "recoupling_distance_kpc": 1e3, "recoupling_time_Myr": 1e4,
                 **_common.light_params(m_g, N_LIGHT,
                                        light_rebuild_interval_Myr=dt_rebuild_Myr)}
    ctx.info.update({"t_dep_Myr": t_dep * units.UNIT_TIME / units.MYR,
                     "t_end_Myr": t_end * units.UNIT_TIME / units.MYR,
                     "m_g_Msun": m_g, "dt_rebuild_Myr": dt_rebuild_Myr})
    return p


def _gas_mass_Msun(ic):
    import h5py
    with h5py.File(ic, "r") as f:
        m = f["PartType0/Masses"][...]
    return float(np.median(m)) * units.UNIT_MASS / units.SOLAR_MASS


def _sfr_integral(path):
    """sum(SFR dt) of the active particles over the run [internal units]."""
    d = np.loadtxt(path, comments="#")
    return float(d[:, 6].sum())


def analyze(ctx):
    checks = []
    tail = _common.light_tail(ctx.path("light.log"))
    if tail is None:
        return [Check("log has a Light totals line", 0, 1, kind="equal")]
    msun = units.SOLAR_MASS / units.UNIT_MASS
    m_g = ctx.info.get("m_g_Msun", 6250.0) * msun
    m_l = m_g / N_LIGHT

    m_sp = tail["m_spawn"] - tail["m_source"]
    acc = m_sp + tail["m_rounding"] + tail["m_held"] + tail["m_to_stars"]
    checks.append(Check("added = spawned + rounding + held + to stars", acc,
                        tail["m_added"], kind="rel", tol=1e-9,
                        source="reservoir mass moves, never created or lost; "
                               "double sums, 10-digit log"))
    sfr_dt = _sfr_integral(ctx.path("SFR.txt"))
    checks.append(Check("added / (eta_M sum SFR dt)", tail["m_added"] / (ETA_M * sfr_dt),
                        1.0, kind="rel", tol=1e-4,
                        source="dm_pending = eta_M SFR dt per active step; SFR.txt "
                               "column 6 printed to 7 digits"))
    sig_round = np.sqrt(max(m_sp, m_l) * m_l)
    checks.append(Check("rounding / added", tail["m_rounding"] / tail["m_added"], 0.0,
                        kind="sigma", tol=3.0, sigma=sig_round / tail["m_added"],
                        source="stochastic rounding of each draw, variance <= "
                               "m_spawned m_l"))

    last = snapshots(ctx.run_dir)[-1]
    n_star = int(last.npart[4])
    m_star = last.read(4, "InitialMasses", cgs=False).sum()
    created = m_sp + tail["m_held"] + tail["m_to_stars"]
    sigma = np.sqrt((ETA_M * m_g) ** 2 * max(n_star, 1) + sig_round ** 2)
    checks.append(Check("spawned + held + to stars / eta_M M_*",
                        created / (ETA_M * m_star), 1.0, kind="sigma", tol=3.0,
                        sigma=sigma / (ETA_M * max(m_star, 1e-30)),
                        source=f"Mdot_w = eta_M SFR; stars Poisson (N_* = {n_star}) "
                               "and the rounding"))
    t_dep_int = ctx.info["t_dep_Myr"] * units.MYR / units.UNIT_TIME
    dt_reb = ctx.info["dt_rebuild_Myr"] * units.MYR / units.UNIT_TIME
    checks.append(Check("left with stars / added", tail["m_to_stars"] / tail["m_added"],
                        3.0 * dt_reb / t_dep_int, kind="max",
                        source="a reservoir holds at most the wind of one rebuild "
                               "interval; a particle forms a star once per t_dep"))

    light = last.read(0, "WindLight", cgs=False) > 0
    masses = last.read(0, "Masses", cgs=False)
    n_light = int(light.sum())
    checks.append(Check("snapshot light count = log spawned", n_light, tail["n_spawn"],
                        kind="equal", source="no merges: clocks frozen while decoupled"))
    checks.append(Check("snapshot light mass / log spawned", masses[light].sum(),
                        tail["m_spawn"], kind="rel", tol=1e-6, source="float masses"))
    checks.append(Check("max |m_light / m_l - 1|", np.abs(masses[light] / m_l - 1).max()
                        if n_light else 1.0, 1e-6, kind="max", source="m_l = m_g / N"))
    dec = last.read(0, "WindDecoupled", cgs=False) > 0
    checks.append(Check("light particles decoupled", int((light & ~dec).sum()), 0,
                        kind="equal", source="recoupling beyond the run"))

    e_w = ETA_E * U_STAR / ETA_M
    u_w = (1.0 - F_KIN) * e_w
    dv = np.sqrt(2.0 * F_KIN * e_w)
    u_launch = last.read(0, "WindLaunchInternalEnergies")[light]
    checks.append(Check("max |u_launch / u_w - 1|", np.abs(u_launch / u_w - 1).max(),
                        1e-6, kind="max", source="u_w = 9/14 eta_E u_* / eta_M"))
    vz = np.abs(last.read(0, "Velocities")[light][:, 2])
    checks.append(Check("mean |v_z| of light particles / Delta v", vz.mean() / dv, 1.0,
                        kind="rel", tol=1e-2,
                        source="Delta v = (2 5/14 eta_E u_* / eta_M)^1/2 added to "
                               "parents moving at a few km/s"))

    import h5py
    with h5py.File(ctx.path("ics.hdf5"), "r") as f:
        m_init = float(f["PartType0/Masses"][...].astype(np.float64).sum())
    m_end = masses.sum() + last.read(4, "Masses", cgs=False).sum()
    checks.append(Check("gas + stars / initial gas", m_end / m_init, 1.0,
                        kind="rel", tol=1e-6, source="float masses"))
    ctx.info.update({"stars": n_star, "light": n_light, "log_tail": tail,
                     "sfr_dt": sfr_dt})
    return checks
