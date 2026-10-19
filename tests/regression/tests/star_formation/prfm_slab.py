"""PRFM-int star formation in a uniform gas layer of known surface density.

A periodic box of 1 x 1 x 0.5 kpc filled with uniform gas is a layer of
surface density Sigma_g = rho L_z seen face-on from the z axis: the face-on
map (src/prfm_map.c, 100 pc pixels, covering the box) sees the same
Sigma_g in every pixel. There are no stars at the start and no dark matter
(PRFMStarFormation:use_dark_matter = 0, constant zero external field), so
J26 eq. 31 reduces to the self-gravity of the gas layer (H24 eq. 5),
    W = pi G Sigma_g^2 / 2,
and the depletion time is
    t_dep = Upsilon(W) Sigma_g / W = 2 Upsilon(W) / (pi G Sigma_g),
with the OK22 yield Upsilon = 1028 km/s (W / 1e4 k_B K cm^-3)^-0.212.
Every particle is above n_th = 0.13 cm^-3 and on or below the effective EoS,
so every particle is star-forming.

Two layers, Sigma_g = 50 and 200 Msun/pc^2 (n_H = 2.9 and 11.6 cm^-3), run for
2 per cent of t_dep. Checks:
  1. stars formed against N_0 (1 - exp(-t / t_dep)), within 4 Poisson sigma.
     Stars formed at rest add to the weight through Sigma_* with
     sigma_*z ~ 0, which raises the rate per gas particle by (1 + f) for a
     converted fraction f; at f <= 2 per cent this shifts the count by
     <= 1 per cent, below the Poisson error (about 4 per cent);
  2. each gas particle's SFR against m_g / t_dep, to 1e-3, at an early
     output (t = 5e-5 t_dep, about one star formed), where Sigma_* is too
     small to change any particle's weight by 1e-3. Later, the stars formed
     make Sigma_* vary from pixel to pixel by tens of per cent, and with it
     each particle's rate by a few per cent, so a per-particle check against
     the mean layer is exact only at the start. The time-step is capped at
     2e-5 t_dep so that every particle has computed its SFR by then;
  3. the fraction of gas particles star-forming is 1.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "PRFM-int: conversion in a uniform gas layer matches t_dep = 2 Upsilon / (pi G Sigma_g)"
TIER = "quick"
BUILDS = {"prfm": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=constant"]}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

# 25 pc lattice: four columns per 100 pc pixel, so every pixel holds the
# same gas mass (a lattice incommensurate with the pixels aliases into
# pixel-to-pixel Sigma_g differences of 10 per cent)
NX, NZ = 40, 20
LX, LZ = 1.0, 0.5  # kpc
SIGMAS = (50.0, 200.0)  # Msun / pc^2
T0 = 1e4  # K
FRACTION = 0.02  # run length, of t_dep
T_EARLY = 5e-5  # early output, of t_dep
DT_MAX = 2e-5  # of t_dep
KMS = 1e5


def upsilon(W):
    """OK22 yield in cm/s for weight W in erg/cm^3."""
    return 1028.0 * KMS * (W / (1e4 * units.BOLTZMANN_K)) ** -0.212


def t_dep_layer(Sigma_g, Sigma_eff=None):
    """Depletion time in s of a gas layer, surface densities in g/cm^2."""
    if Sigma_eff is None:
        Sigma_eff = Sigma_g
    W = 0.5 * np.pi * units.G_NEWTON * Sigma_g * Sigma_eff
    return upsilon(W) * Sigma_g / W


def lattice():
    dx = LX / NX
    x = (np.arange(NX) + 0.5) * dx
    z = (np.arange(NZ) + 0.5) * (LZ / NZ)
    xx, yy, zz = np.meshgrid(x, x, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def write_layer(path, Sigma_msun_pc2):
    npart = NX * NX * NZ
    Sigma = Sigma_msun_pc2 * units.SOLAR_MASS / units.PARSEC ** 2
    mass_cgs = Sigma * (LX * units.KPC) ** 2 / npart
    u = units.u_from_T(T0, units.MU_NEUTRAL) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": lattice(),
        "Velocities": np.zeros((npart, 3)),
        "Masses": np.full(npart, mass_cgs / units.UNIT_MASS),
        "SmoothingLength": np.full(npart, 1.2348 * LX / NX),
        "InternalEnergy": np.full(npart, u),
        "ParticleIDs": np.arange(1, npart + 1, dtype=np.uint64),
    }
    ics.write(path, np.array([LX, LX, LZ]), gas=gas)
    return Sigma


def run(ctx):
    ctx.table("photometry")
    ctx.table("coolingtables")
    for S in SIGMAS:
        name = f"sigma{S:g}"
        os.makedirs(ctx.path(name))
        Sigma = write_layer(ctx.path(name, "ics.hdf5"), S)
        t_dep = t_dep_layer(Sigma) / units.UNIT_TIME
        t_end = FRACTION * t_dep
        p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, t_end,
                              dt_max=DT_MAX * t_dep)
        with open(ctx.path(name, "output_list.txt"), "w") as f:
            f.write(f"# Time\n{T_EARLY * t_dep!r}\n{t_end!r}\n")
        p["Snapshots"].update({"output_list_on": 1, "output_list": "output_list.txt"})
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
        p["SPH"]["h_max"] = 2.0 * 1.2348 * LX / NX
        p["PRFMEntropyFloor"] = {"calibration": "OK22",
                                 "density_threshold_H_p_cm3": 0.13}
        p["PRFMStarFormation"] = {
            "calibration": "OK22", "threshold": "measured",
            "threshold_n_H_p_cm3": 0.13, "use_dark_matter": 0,
            "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
            "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
            "map_half_width_kpc": 0.5 * LX}
        ctx.info[name] = {"t_dep_Myr": t_dep_layer(Sigma) / units.MYR,
                          "t_end_Myr": t_end * units.UNIT_TIME / units.MYR}
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=1800)


def analyze(ctx):
    checks = []
    area = (LX * units.KPC) ** 2
    n0 = NX * NX * NZ
    for S in SIGMAS:
        name = f"sigma{S:g}"
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        Sigma0 = S * units.SOLAR_MASS / units.PARSEC ** 2
        t_dep0 = t_dep_layer(Sigma0)

        nstar = last.npart[4]
        expected = n0 * (1.0 - np.exp(-last.time_cgs / t_dep0))
        checks.append(Check(f"Sigma_g = {S:g}: stars formed", nstar, expected,
                            kind="sigma", tol=4.0, sigma=np.sqrt(expected),
                            source=f"N_0 (1 - exp(-t/t_dep)), t_dep = 2 Upsilon / "
                                   f"(pi G Sigma_g) = {t_dep0 / units.MYR:.1f} Myr; "
                                   "Poisson sigma"))

        early = snaps[0]
        m = early.read(0, "Masses")
        sfr = early.read(0, "StarFormationRates")
        ms = early.read(4, "Masses")
        Sg = m.sum() / area
        Ss = ms.sum() / area if ms.size else 0.0
        t_dep = t_dep_layer(Sg, Sg + 2.0 * Ss)
        rel = np.abs(sfr * t_dep / m - 1.0)
        checks.append(Check(f"Sigma_g = {S:g}: fraction star-forming",
                            (sfr > 0).mean(), 1.0, kind="equal",
                            source="n_H > 0.13 cm^-3, on or below the floor"))
        checks.append(Check(f"Sigma_g = {S:g}: max |SFR t_dep / m_g - 1|, early",
                            rel.max(), 1e-3, kind="max",
                            source=f"t_dep = 2 Upsilon(W) / (pi G Sigma_eff), "
                                   f"W = pi G Sigma_g Sigma_eff / 2, Sigma_eff = Sigma_g + "
                                   f"2 Sigma_*, at t = {early.time_cgs / units.MYR:.3g} "
                                   f"Myr ({early.npart[4]} stars)"))
        ctx.info[name].update({"stars": int(nstar), "expected": expected,
                               "converted_fraction": nstar / n0,
                               "max_rel_sfr": rel.max()})
    return checks
