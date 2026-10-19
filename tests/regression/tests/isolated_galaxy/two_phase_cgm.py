"""Milky Way-mass disc in a hot CGM with the two-phase wind and CR-carrying wind
particles (Experiment B, wind-design.md Sec. 10; long tier).

ICs: the H12 disc with a hot CGM (swift-ism-model add_cgm.py),
<data>/ics/<RES>_cgm_disk.hdf5, RES = H12_m1.6e6 (default; softening 0.5 kpc) or
H12_m1.3e7 (SWIFT_REG_ISOGAL_RES). Set-up of isolated_galaxy/prfm_wind
(the Hernquist halo of the H12 tests, PRFM-int with OK22, TIGRESS effective EoS
floor, EAGLE cooling, launch along +-z) with the gas
metals of the ICs, SPH and Stars h_max = 150 kpc and 8 top-level cells (box
4565 kpc: cell 570 kpc, above 1.1 kernel_gamma h_max = 301 kpc). Recoupling at
max(2 kpc, 2 kernel supports) or 25 Myr (hot), max(1.5 kpc, 2 kernel
supports), 25 Myr or the apex (warm). Runs (SWIFT_REG_2PH_RUNS, default all):
  nowind, tig_hot, tig_2ph, tig_2ph_cr, tig_2ph_crnone
with the winds of isolated_galaxy/arkenstone_twophase. Length
SWIFT_REG_2PH_TEND_MYR (default 1000), snapshots every 10 Myr.

Checks: each run reached its end time; the loading and CR bookkeeping checks of
arkenstone_twophase (sum m/eta per phase against the stellar mass formed within
3 sigma; CR launched = released + carried to 1e-6; deposited as configured).
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import prfm_disc
from . import arkenstone_twophase as ark2

DESCRIPTION = "H12 disc in a hot CGM with the two-phase wind and CR-carrying wind particles"
TIER = "long"
BUILDS = {"wind": prfm_disc.BUILDS["prfm"] + ["--with-wind=TIGRESS"]}

RES = os.environ.get("SWIFT_REG_ISOGAL_RES", "H12_m1.6e6")
SOFTENING = {"H12_m1.6e6": 0.5, "H12_m1.3e7": 1.0}[RES]
IC_NAME = f"{RES}_cgm_disk.hdf5"
MYR = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_2PH_TEND_MYR", "1000"))
H_MAX = 150.0  # kpc

WINDS = dict({"nowind": {"loading_model": "none"}},
             **{k: ark2.WINDS[k] for k in ("tig_hot", "tig_2ph", "tig_2ph_cr",
                                           "tig_2ph_crnone")})
_only = os.environ.get("SWIFT_REG_2PH_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]
FLAGS = ["--hydro", "--self-gravity", "--external-gravity", "--stars",
         "--star-formation", "--cooling", "--limiter", "--sync"]


def disc_params(ctx):
    p = params.eagle_base(IC_NAME, T_END_MYR * MYR, ctx.data_dir, 10.0 * MYR,
                          dt_max=1e-2, cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
    p["Snapshots"]["compression"] = 4
    p["Gravity"].update({"max_physical_DM_softening": SOFTENING,
                         "max_physical_baryon_softening": SOFTENING})
    p["SPH"].update({"h_min_ratio": 0.1, "h_max": H_MAX, "minimal_temperature": 100.0})
    p["Stars"]["h_max"] = H_MAX
    p["EAGLEChemistry"]["init_abundance_metal"] = -1
    p["Scheduler"] = {"max_top_level_cells": 8, "tasks_per_cell": 200}
    p["HernquistPotential"] = {
        "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
        "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
        "bulgefraction": 0.014, "timestep_mult": 0.01, "epsilon": SOFTENING}
    p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
    p["PRFMStarFormation"] = dict(
        calibration="OK22", threshold="measured", threshold_n_H_p_cm3=0.13,
        use_dark_matter=1, disc_centre=[0.0, 0.0, 0.0], disc_centre_is_absolute=0,
        pixel_kpc=0.1, cylinder_radius_kpc=0.5, half_height_kpc=1.0,
        map_half_width_kpc=25.0)
    return p


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    path = os.path.join(ctx.data_dir, "ics", IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: make it with swift-ism-model add_cgm.py")
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(path, ctx.path(name, IC_NAME))
        p = disc_params(ctx)
        p["Wind"] = dict(WINDS[name], launch_direction=[0.0, 0.0, 1.0],
                         recoupling_distance_kpc=2.0, recoupling_kernel_factor=2.0,
                         recoupling_time_Myr=25.0, max_timestep_Myr=1.0)
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=24 * 3600)


def analyze(ctx):
    checks = []
    for name in RUNS:
        last = snapshots(ctx.path(name))[-1]
        checks.append(Check(f"{name}: end time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-3,
                            source="SWIFT_REG_2PH_TEND_MYR"))
        if name != "nowind":
            ctx.info[name] = ark2.wind_checks(ctx, name, WINDS[name], last, checks)
    return checks
