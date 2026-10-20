"""Milky Way-mass isolated disc at near-target resolution, f_kin series (long tier).

Chaikin et al. (2023) ran isolated discs to see what the thermal-kinetic scheme does
to a galaxy: its star formation history, structure, outflows and the
Kennicutt-Schmidt relation. This test repeats their H12 set-up with SWIFT's
EAGLE model and the EAGLE-thermal-kinetic module.

ICs: the COLIBRE isolated disc M6_disk of Nobels et al. (2024; SWIFT web storage,
ICs/IsolatedGalaxies_COLIBRE), gas particle mass 8e5 Msun (Chaikin+23 "M6"), 20550
gas and 47950 star particles, in a Hernquist potential with M200 = 1.37e12 Msun,
c = 9, disc fraction 0.04, bulge fraction 0.014 (as Chaikin+23 Sec. 3.2 and the SWIFT
IsolatedGalaxy_feedback example). Softening 0.4 kpc. EAGLE star formation (pressure
law), cooling and reference entropy floor; enrichment on, SNIa continuous, with the
COLIBRE delay-time distribution exp(-(t - t_d)/tau).
SN feedback as Chaikin+23's fiducial: f_E = 2, Delta T = 10^7.5 K, Delta v = 50 km/s,
8 rays, with f_kin = 0, 0.1, 1, and one run with COLIBRE's Delta T(n_H) (m6 values)
at f_kin = 0.1. 1 Gyr, snapshots every 10 Myr.

Recorded: star formation rate history (from star birth times), and the snapshots for
maps, profiles, the Kennicutt-Schmidt relation and outflow rates (analysed in
swift-ism-model). Check: each run reached 1 Gyr.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "isolated MW-mass disc (M6 or M5): f_kin = 0, 0.1, 1 and COLIBRE Delta T(n)"
TIER = "long"
# Hydro scheme from SWIFT_REG_HYDRO (default SPHENIX), as in sn_blast/single_sn_cooling.
HYDRO = os.environ.get("SWIFT_REG_HYDRO", "sphenix")
BUILDS = {"tk": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
                 "--with-entropy-floor=EAGLE", "--with-stars=EAGLE",
                 "--with-star-formation=EAGLE", "--with-black-holes=EAGLE",
                 "--with-ext-potential=hernquist", "--with-number-of-SNII-rays=8",
                 "--with-feedback=EAGLE-thermal-kinetic"]
                + ([] if HYDRO == "sphenix" else
                   [f"--with-hydro={HYDRO}"]
                   + (["--with-riemann-solver=hllc"] if HYDRO.startswith("gizmo") else []))}

# Resolution and run subset from the environment, so that each run can be its
# own cluster job: SWIFT_REG_ISOGAL_RES = M6 (default, 8e5 Msun, softening 0.4 kpc)
# or M5 (1e5 Msun, 0.2 kpc), or H12_m3e6 / H12_m3e7 (M6_disk subsampled to 3.2e6 and
# 2.6e7 Msun by swift-ism-model tests/isolated_galaxy/make_ics.py; 0.8 and
# 1.7 kpc), or H12_m1.6e6 / H12_m1.3e7, the two reference resolutions (HR5/8 and
# HR5, Lee et al. 2021: baryon mass 1.6e6 and 1.3e7 Msun; 0.5 and 1.0 kpc, HR5 has
# 1 kpc); SWIFT_REG_ISOGAL_RUNS = comma-separated names. A suffix _cgm (H12_m1.6e6_cgm)
# selects the same disc inside a hydrostatic hot CGM, made by swift-ism-model
# tests/isolated_galaxy/add_cgm.py.
RES = os.environ.get("SWIFT_REG_ISOGAL_RES", "M6")
SOFTENING = {"M5": 0.2, "M6": 0.4, "H12_m3e6": 0.8, "H12_m3e7": 1.7,
             "H12_m1.6e6": 0.5, "H12_m1.3e7": 1.0}[RES.replace("_cgm", "")]
IC_NAME = f"{RES}_disk.hdf5"
IC_URL = ("https://virgodb.cosma.dur.ac.uk/swift-webstorage/ICs/IsolatedGalaxies_COLIBRE/"
          + IC_NAME)
T_END = 1.0227  # internal units (kpc / (km/s)), 1 Gyr
DT_SNAP = 1.0227e-2  # 10 Myr
COLIBRE_M6 = {"SNII_delta_T_model": "DensityDependent", "SNII_delta_T_K": 10 ** 6.5,
              "SNII_delta_T_n_pivot_H_p_cm3": 0.5, "SNII_delta_T_min_K": 10 ** 6.75,
              "SNII_delta_T_max_K": 1e8}
ALL_RUNS = {
    "fkin0": {"SNII_f_kinetic": 0.0},
    "fkin0.1": {"SNII_f_kinetic": 0.1},
    "fkin1": {"SNII_f_kinetic": 1.0},
    "colibre": dict(COLIBRE_M6, SNII_f_kinetic=0.1),
    # Referee control (ai-notes referee-ism-model-notes.md, Blocking 1): stochastic heating
    # set to the TIGRESS hot loadings, eta_M,hot = 0.2 and eta_E,hot = 0.1 per 1e51 erg per
    # 95.5 Msun: f_E = 0.1 / (0.0118 x 95.5) = 0.089 and Delta T from
    # eta_M = 0.91 f_E (Delta T / 10^7.5 K)^-1 -> Delta T = 10^7.11 K.
    "tigress_heat": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 10 ** 7.11,
                     "SNII_energy_fraction_min": 0.089, "SNII_energy_fraction_max": 0.089},
    # no cooling, star formation or feedback: tests that the ICs (disc and CGM) are
    # in equilibrium
    "adiabatic": {},
    # baseline without supernova energy (SNIa feedback off as well)
    "nofb": {"SNII_energy_fraction_min": 0.0, "SNII_energy_fraction_max": 0.0,
             "use_SNIa_feedback": 0},
}
_only = os.environ.get("SWIFT_REG_ISOGAL_RUNS")
RUNS = {k: v for k, v in ALL_RUNS.items() if not _only or k in _only.split(",")}


def ic_file(data_dir):
    import subprocess
    path = os.path.join(data_dir, "ics", IC_NAME)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-o", path + ".part", IC_URL], check=True)
        os.replace(path + ".part", path)
    return path


def run(ctx):
    for t in ("coolingtables", "photometry", "yieldtables"):
        ctx.table(t)
    ics = ic_file(ctx.data_dir)
    for name, fb in RUNS.items():
        os.makedirs(ctx.path(name))
        shutil.copy(ics, ctx.path(name, IC_NAME))
        p = params.eagle_base(IC_NAME, T_END, ctx.data_dir, DT_SNAP, dt_max=1e-2)
        p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
        p["Snapshots"]["compression"] = 4
        p["Gravity"].update({"max_physical_DM_softening": SOFTENING,
                             "max_physical_baryon_softening": SOFTENING})
        # h_max 10 kpc (the COLIBRE example value) leaves particles in the hot CGM and in
        # outflows with too few neighbours; SPHENIX only warns, GIZMO's volume estimate
        # degenerates and the time-step collapses. SWIFT_REG_ISOGAL_HMAX overrides it; keep
        # it below the top-level cell size (box / max_top_level_cells = 285 kpc here).
        p["SPH"].update({"h_min_ratio": 0.1,
                         "h_max": float(os.environ.get("SWIFT_REG_ISOGAL_HMAX", "10")),
                         "minimal_temperature": 100.0})
        # the kernel support (1.83 h_max) must fit in a top-level cell with margin; for CGM
        # runs use SWIFT_REG_ISOGAL_TOPCELLS = 8 (570 kpc cells) with h_max 150
        p["Scheduler"] = {"max_top_level_cells": int(os.environ.get("SWIFT_REG_ISOGAL_TOPCELLS", "16")),
                          "tasks_per_cell": 200}
        p["HernquistPotential"] = {
            "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
            "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
            "bulgefraction": 0.014, "timestep_mult": 0.01, "epsilon": SOFTENING}
        p["EAGLEFeedback"].update({
            "SNII_energy_fraction_min": 2.0, "SNII_energy_fraction_max": 2.0,
            "SNII_delta_T_K": 10 ** 7.5, "SNII_delta_v_km_p_s": 50.0,
            "SNII_sampled_delay": 1, "SNIa_DTD_convention": "COLIBRE"})
        p["EAGLEFeedback"].update(fb)
        if RES.endswith("_cgm"):  # per-particle metals from the ICs (CGM at 0.3 solar)
            p["EAGLEChemistry"]["init_abundance_metal"] = -1
        physics = (["--temperature"] if name == "adiabatic" else
                   ["--star-formation", "--cooling", "--feedback"])
        ctx.swift(p, ["--hydro", "--self-gravity", "--external-gravity", "--stars"]
                  + physics + ["--limiter", "--sync"],
                  subdir=name, name=name, timeout=6 * 3600)


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        birth = last.read(4, "BirthTimes") / units.MYR
        m = last.read(4, "InitialMasses") / units.SOLAR_MASS
        new = birth > 0
        edges = np.arange(0, 1001, 10.0)
        sfr = np.histogram(birth[new], edges, weights=m[new])[0] / 10e6
        ctx.info[name] = {"t_Myr": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                          "sfr_Msun_yr": sfr.tolist()}
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR, 1000.0,
                            kind="rel", tol=1e-2, source="run completed"))
    return checks
