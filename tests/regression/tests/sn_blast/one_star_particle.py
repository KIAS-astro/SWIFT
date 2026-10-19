"""One star particle at the two reference resolutions, with snapshots kept (long).

Companion to target_resolution, for slices, movies and the evolution of what one star
particle does to its surroundings at galaxy-formation resolution: m_g = 1.6e6 Msun
(eight times finer than Horizon Run 5) and 1.3e7 Msun (Horizon Run 5), uniform gas at
n_H = 1 cm^-3, solar metallicity, EAGLE cooling and entropy floor, 32^3 particles on a
glass (boxes of 12 and 24 kpc), one star particle of the gas mass at the centre born at
t = 0 releasing its CCSNe with sampled delays (f_E = 1; SNIa and enrichment off), to
50 Myr, outputs every 0.5 Myr. Schemes as target_resolution: none, kin (f_kin = 1,
50 km/s), stoch (f_kin = 0, Delta T = 10^7.5 K), colibre (f_kin = 0.1, Delta T(n_H),
m6 values). The stochastic schemes use the star ID of realisation 1 of
target_resolution, which heats at n_H = 1 at 1.6e6 Msun. Check: runs completed.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from .target_resolution import BUILDS, COLIBRE_M6, N, T_INIT  # noqa: F401

DESCRIPTION = "one star particle at 1.6e6 and 1.3e7 Msun, n_H = 1, snapshots every 0.5 Myr"
TIER = "long"
MASSES = {"m1.6e6": 1.6e6, "m1.3e7": 1.3e7}
NH = 1.0
SCHEMES = {
    "none": None,
    "kin": {"SNII_f_kinetic": 1.0, "SNII_delta_v_km_p_s": 50.0},
    "stoch": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 10 ** 7.5},
    "colibre": dict(COLIBRE_M6, SNII_f_kinetic=0.1, SNII_delta_v_km_p_s=50.0),
}
DT_SNAP = 0.5113e-3  # 0.5 Myr
T_END = 5.113e-2  # 50 Myr
STAR_ID_OFFSET = 7919  # realisation 1 of target_resolution


def box_kpc(m):
    return (N ** 3 * m * units.SOLAR_MASS / units.rho_from_nH(NH)) ** (1 / 3) / units.KPC


def run(ctx):
    from swiftreg import data
    for t in ("coolingtables", "photometry", "yieldtables"):
        ctx.table(t)
    glass = data.glass(32, ctx.data_dir, log=ctx.log)
    for mname, m in MASSES.items():
        L = box_kpc(m)
        for scheme, fb in SCHEMES.items():
            name = f"{scheme}_{mname}"
            os.makedirs(ctx.path(name))
            gas, _ = ics.uniform_gas(N, L, NH, T_INIT, glass_file=glass)
            stars = {
                "Coordinates": np.full((1, 3), 0.5 * L),
                "Velocities": np.zeros((1, 3)),
                "Masses": np.array([m * units.SOLAR_MASS / units.UNIT_MASS]),
                "SmoothingLength": np.array([1.2348 * L / N]),
                "ParticleIDs": np.array([N ** 3 + 1 + STAR_ID_OFFSET], dtype=np.uint64),
            }
            ics.write(ctx.path(name, "ics.hdf5"), L, gas=gas, stars=stars)
            p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-3)
            p["Snapshots"]["compression"] = 4
            p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
            p["Stars"].update({"overwrite_birth_time": 1, "birth_time": 0.0})
            p["SPH"]["h_max"] = L / 8
            p["EAGLEFeedback"].update({
                "use_SNIa_feedback": 0, "use_AGB_enrichment": 0, "use_SNII_enrichment": 0,
                "use_SNIa_enrichment": 0, "AGB_ejecta_velocity_km_p_s": 0.0,
                "SNII_energy_fraction_min": 1.0, "SNII_energy_fraction_max": 1.0,
                "SNII_sampled_delay": 1,
            })
            flags = ["--hydro", "--stars", "--external-gravity", "--cooling", "--limiter",
                     "--sync"]
            if fb is not None:
                p["EAGLEFeedback"].update(fb)
                flags.append("--feedback")
            ctx.info[name] = {"box_kpc": L, "m_gas_Msun": float(gas["Masses"][0] * units.UNIT_MASS
                                                                 / units.SOLAR_MASS)}
            ctx.swift(p, flags, subdir=name, name=name, timeout=7200)


def analyze(ctx):
    checks = []
    for mname in MASSES:
        for scheme in SCHEMES:
            name = f"{scheme}_{mname}"
            last = snapshots(ctx.path(name))[-1]
            checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                                T_END * units.UNIT_TIME / units.MYR, kind="rel", tol=1e-3,
                                source="run completed"))
    return checks
