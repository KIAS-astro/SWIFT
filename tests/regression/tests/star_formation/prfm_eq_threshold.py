"""PRFM equilibrium threshold: diffuse gas above the plane forms no stars.

With PRFMStarFormation:threshold = equilibrium a particle is eligible when the
equilibrium density of its column, n_H,eq = W / (sigma_eff^2 1.4 m_H), is at
least n_th = 0.13 cm^-3, and its own measured n_H is at least the floor
equilibrium_min_measured_n_H_p_cm3 (J26 p. 23; default 0.01 cm^-3). The
column is read from the face-on map at the particle's (x, y) whatever its
height, so without the floor every particle above a dense disc qualifies.

A periodic box of 5 x 5 x 7 kpc holds
  a layer of Sigma_g = 50 Msun/pc^2, 250 pc thick, at the centre: measured
      n_H = 5.8 cm^-3, n_H,eq = 1.3 cm^-3 (W = pi G Sigma_g^2 / 2,
      sigma_eff = 12 km/s (W/P_0)^0.22 = 19 km/s);
  2 x 900 diffuse particles of 1/8 the layer particle mass on a 333 pc
      lattice at 1.5-2.5 kpc above and below it: measured n_H = 0.0024
      cm^-3, outside the map slab (|z| < 1 kpc), so
      they add nothing to Sigma_g, and more than one kernel support away from
      the layer, so the layer does not raise their SPH density. Their column
      is the layer's, n_H,eq = 1.3 cm^-3.
The diffuse particles start with h = 0.7 kpc, above their converged value, so
that the top-level cells (at least three across, each wider than the
largest kernel support) never become fewer, which SWIFT does not handle.
All gas is at 1e4 K (below T_max = 1e5 K) with no cooling, run for 1 Myr.
Runs and checks, on the star formation rates at 1 Myr:
  floor     default floor 0.01 cm^-3: every layer particle star-forming,
            no diffuse particle star-forming;
  no_floor  floor 0: every particle star-forming, so the floor alone excludes
            the diffuse gas in the first run.
The set-up itself is checked from the written fields: the diffuse gas has
measured n_H below 0.01 and n_H,eq above 0.13 cm^-3. All checks are exact
counts.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "PRFM equilibrium threshold excludes diffuse gas below the measured floor"
TIER = "quick"
BUILDS = {"prfm": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=constant"]}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

LX, LZ = 5.0, 7.0  # kpc
NX, NZ = 100, 5  # layer lattice, 50 pc
SIGMA = 50.0  # Msun / pc^2
ND = 15  # diffuse lattice per side, 333 pc
Z_DIFFUSE = 1.5 + np.arange(4) * LX / ND  # kpc from the midplane
H_DIFFUSE = 0.7  # kpc
M_DIFFUSE = 1.0 / 8.0  # of the layer particle mass
T0 = 1e4  # K
T_END_MYR = 1.0
N_TH = 0.13
FLOOR = 0.01
RUNS = {"floor": None, "no_floor": 0.0}


def write(path):
    dx = LX / NX
    x = (np.arange(NX) + 0.5) * dx
    z = 0.5 * LZ + (np.arange(NZ) - 0.5 * (NZ - 1)) * dx
    xx, yy, zz = np.meshgrid(x, x, z, indexing="ij")
    layer = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    xd = (np.arange(ND) + 0.5) * (LX / ND)
    zd = np.concatenate([0.5 * LZ + Z_DIFFUSE, 0.5 * LZ - Z_DIFFUSE])
    xx, yy, zz = np.meshgrid(xd, xd, zd, indexing="ij")
    diffuse = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    pos = np.vstack([layer, diffuse])
    n = len(pos)
    mass_cgs = (SIGMA * units.SOLAR_MASS / units.PARSEC ** 2
                * (LX * units.KPC) ** 2 / len(layer))
    h = np.concatenate([np.full(len(layer), 1.2348 * dx),
                        np.full(len(diffuse), H_DIFFUSE)])
    u = units.u_from_T(T0, units.MU_NEUTRAL) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": pos,
        "Velocities": np.zeros((n, 3)),
        "Masses": np.concatenate([
            np.full(len(layer), mass_cgs),
            np.full(len(diffuse), M_DIFFUSE * mass_cgs)]) / units.UNIT_MASS,
        "SmoothingLength": h,
        "InternalEnergy": np.full(n, u),
        "ParticleIDs": np.arange(1, n + 1, dtype=np.uint64),
    }
    ics.write(path, np.array([LX, LX, LZ]), gas=gas)
    return len(layer)


def run(ctx):
    ctx.table("photometry")
    ctx.table("coolingtables")
    t_end = T_END_MYR * units.MYR / units.UNIT_TIME
    for name, floor in RUNS.items():
        os.makedirs(ctx.path(name))
        write(ctx.path(name, "ics.hdf5"))
        p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, t_end,
                              dt_max=0.1 * t_end)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
        p["SPH"]["h_max"] = 1.0
        p["PRFMEntropyFloor"] = {"calibration": "OK22",
                                 "density_threshold_H_p_cm3": N_TH}
        sf = {
            "calibration": "OK22", "threshold": "equilibrium",
            "threshold_n_H_p_cm3": N_TH, "use_dark_matter": 0,
            "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
            "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
            "map_half_width_kpc": 0.5 * LX}
        if floor is not None:
            sf["equilibrium_min_measured_n_H_p_cm3"] = floor
        p["PRFMStarFormation"] = sf
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=1800)


def analyze(ctx):
    checks = []
    for name in RUNS:
        last = snapshots(ctx.path(name))[-1]
        z = last.read(0, "Coordinates", cgs=False)[:, 2] - 0.5 * LZ
        diffuse = np.abs(z) > 1.0
        sf = last.read(0, "StarFormationRates") > 0
        n_meas = last.read(0, "Densities") / (1.4 * units.PROTON_MASS)
        n_eq = last.read(0, "PRFMEquilibriumDensities") / (1.4 * units.PROTON_MASS)
        src = f"{name}: threshold equilibrium, n_th = {N_TH}"
        checks.append(Check(f"{name}: diffuse particles", diffuse.sum(),
                            2 * len(Z_DIFFUSE) * ND ** 2,
                            kind="equal", source="none converted or moved off"))
        checks.append(Check(f"{name}: max measured n_H of diffuse gas [cm^-3]",
                            n_meas[diffuse].max(), FLOOR, kind="max",
                            source="set-up: diffuse gas below the floor"))
        checks.append(Check(f"{name}: min n_H,eq of diffuse gas [cm^-3]",
                            n_eq[diffuse].min(), N_TH, kind="min",
                            source="set-up: the column passes the equilibrium cut"))
        checks.append(Check(f"{name}: fraction of layer star-forming",
                            sf[~diffuse].mean(), 1.0, kind="equal",
                            source=src + "; measured and equilibrium n_H pass"))
        expect = 0.0 if RUNS[name] is None else 1.0
        checks.append(Check(f"{name}: fraction of diffuse gas star-forming",
                            sf[diffuse].mean(), expect, kind="equal",
                            source=src + (f"; measured floor {FLOOR} cm^-3"
                                          if RUNS[name] is None else "; floor 0")))
        ctx.info[name] = {"n_meas_diffuse": [n_meas[diffuse].min(),
                                             n_meas[diffuse].max()],
                          "n_eq_diffuse_min": n_eq[diffuse].min(),
                          "n_meas_layer_median": np.median(n_meas[~diffuse])}
    return checks
