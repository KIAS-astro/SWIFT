"""Milky Way-mass isolated disc with PRFM-int star formation and the
decoupled TIGRESS wind: a smoke test (long tier).

Set-up of isolated_galaxy/prfm_disc (H12 disc, Hernquist halo, PRFM-int with
OK22 and the TIGRESS effective EoS, EAGLE cooling) with --with-wind=TIGRESS:
star-forming particles launch with the K20b hot-phase loadings at their
Sigma_SFR = W / Upsilon_tot along +-z, and recouple at max(2 kpc, 2 kernel
supports) or after 25 Myr (wind-design.md, WD1-WD7). Default length 100 Myr
(SWIFT_REG_WIND_TEND_MYR), snapshots every 10 Myr.

Checks that winds launch, travel and recouple:
  1. the run reached its end time;
  2. particles launched, and launches / stars formed lies within 3 Poisson
     sigma of the range of eta_M,hot over Sigma_SFR in [1e-4, 1]
     (0.16-0.26);
  3. particles recoupled, each at or beyond its recoupling distance or
     after t_rec;
  4. decoupled particles at the end have moved away from the disc plane:
     median |z - z_launch| of the decoupled ones > 0.2 kpc;
  5. the log's wind totals agree with the snapshot (launch count).
Recorded: launched mass and energy, energy at recoupling, launch and
recoupling heights.
"""

import os
import shutil

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import prfm_disc
from ..wind import _common as wind_common

DESCRIPTION = "isolated MW-mass disc, PRFM-int SF with the decoupled TIGRESS wind (smoke test)"
TIER = "long"
BUILDS = {"wind": prfm_disc.BUILDS["prfm"] + ["--with-wind=TIGRESS"]}

MYR_INTERNAL = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_WIND_TEND_MYR", "100"))
ETA_RANGE = (10 ** -0.86 * 1.0 ** -0.07, 10 ** -0.86 * 1e-4 ** -0.07)


def run(ctx):
    for t in ("coolingtables", "photometry"):
        ctx.table(t)
    ics = prfm_disc.ic_file(ctx.data_dir)
    shutil.copy(ics, ctx.path(prfm_disc.IC_NAME))
    # Reuse the prfm_disc parameters by building them the same way
    p = _disc_params(ctx)
    p["Wind"] = {"loading_model": "TIGRESS", "launch_direction": [0.0, 0.0, 1.0],
                 "recoupling_distance_kpc": 2.0, "recoupling_kernel_factor": 2.0,
                 "recoupling_time_Myr": 25.0, "max_timestep_Myr": 1.0}
    ctx.swift(p, ["--hydro", "--self-gravity", "--external-gravity", "--stars",
                  "--star-formation", "--cooling", "--limiter", "--sync"],
              name="wind", timeout=12 * 3600)


def _disc_params(ctx):
    from swiftreg import params
    soft = prfm_disc.SOFTENING
    p = params.eagle_base(prfm_disc.IC_NAME, T_END_MYR * MYR_INTERNAL, ctx.data_dir,
                          10.0 * MYR_INTERNAL, dt_max=1e-2)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
    p["Snapshots"]["compression"] = 4
    p["Gravity"].update({"max_physical_DM_softening": soft,
                         "max_physical_baryon_softening": soft})
    p["SPH"].update({"h_min_ratio": 0.1, "h_max": 10.0, "minimal_temperature": 100.0})
    p["Scheduler"] = {"max_top_level_cells": 16, "tasks_per_cell": 200}
    p["HernquistPotential"] = {
        "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
        "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
        "bulgefraction": 0.014, "timestep_mult": 0.01, "epsilon": soft}
    p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
    p["PRFMStarFormation"] = dict(
        calibration="OK22", threshold="measured", threshold_n_H_p_cm3=0.13,
        use_dark_matter=1, disc_centre=[0.0, 0.0, 0.0], disc_centre_is_absolute=0,
        pixel_kpc=0.1, cylinder_radius_kpc=0.5, half_height_kpc=1.0,
        map_half_width_kpc=25.0)
    return p


def analyze(ctx):
    checks = []
    snaps = snapshots(ctx.run_dir)
    last = snaps[-1]
    checks.append(Check("final time [Myr]", last.time_cgs / units.MYR, T_END_MYR,
                        kind="rel", tol=1e-2, source="run completed"))
    birth = last.read(4, "BirthTimes")
    n_star = int((birth > 0).sum())
    counts = last.read(0, "WindLaunchCounts", cgs=False)
    n_w = int(counts.sum())
    checks.append(Check("launches", n_w, 1, kind="min", source="PRFM SF with wind"))
    ratio = n_w / max(n_star, 1)
    lo = ETA_RANGE[0] - 3 * np.sqrt(ETA_RANGE[0] * 1.3 * max(n_star, 1)) / max(n_star, 1)
    hi = ETA_RANGE[1] + 3 * np.sqrt(ETA_RANGE[1] * 1.3 * max(n_star, 1)) / max(n_star, 1)
    checks.append(Check("launches / stars formed >= eta_M,min - 3 sigma", ratio, lo,
                        kind="min", source=f"eta_M,hot in {ETA_RANGE[0]:.3f}-"
                                           f"{ETA_RANGE[1]:.3f}; N_* = {n_star}"))
    checks.append(Check("launches / stars formed <= eta_M,max + 3 sigma", ratio, hi,
                        kind="max", source="as above"))

    dec = last.read(0, "WindDecoupled", cgs=False) > 0
    t_rec = last.read(0, "WindRecouplingTimes")
    rec = t_rec > 0
    d = last.read(0, "WindRecouplingDistances")[rec]
    d_thr = last.read(0, "WindRecouplingDistanceThresholds")[rec]
    t_l = last.read(0, "WindLaunchTimes")[rec]
    by_d = d >= d_thr * (1 - 1e-6)
    by_t = (t_rec[rec] - t_l) >= 25.0 * units.MYR * (1 - 1e-6)
    checks.append(Check("recoupled", int(rec.sum()), 1, kind="min",
                        source="distance or time reached"))
    checks.append(Check("recoupled by distance or time (fraction)",
                        (by_d | by_t).mean() if rec.any() else 0.0, 1.0, kind="equal",
                        source="WD5"))
    dx = last.read(0, "WindDisplacements")[dec]
    dz = np.abs(dx[:, 2]) / units.KPC if dec.any() else np.zeros(1)
    checks.append(Check("decoupled: median |dz| since launch [kpc]", np.median(dz), 0.2,
                        kind="min", source="kick along +-z of a few hundred km/s"))
    tail = wind_common.wind_tail(ctx.path("wind.log"))
    checks.append(Check("log: launches", tail["n_launch"] if tail else -1, n_w,
                        kind="equal", source="Wind totals at the last statistics"))

    m = last.read(0, "Masses")
    ek0 = last.read(0, "WindLaunchKineticEnergies")
    eth0 = last.read(0, "WindLaunchInternalEnergies")
    ek1 = last.read(0, "WindRecouplingKineticEnergies")
    eth1 = last.read(0, "WindRecouplingInternalEnergies")
    launched = counts > 0
    z_launch = last.read(0, "WindLaunchPositions")[launched][:, 2] / units.KPC
    ctx.info.update({
        "stars": n_star, "launches": n_w, "recoupled": int(rec.sum()),
        "decoupled_at_end": int(dec.sum()),
        "m_launched_Msun_last": float(m[launched].sum() / units.SOLAR_MASS),
        "e_launch_mean_kms2": float(np.mean((ek0 + eth0)[launched]) / 1e10) if launched.any() else 0,
        "e_recouple_mean_kms2": float(np.mean((ek1 + eth1)[rec]) / 1e10) if rec.any() else 0,
        "median_recouple_distance_kpc": float(np.median(d) / units.KPC) if rec.any() else 0,
        "median_d_rec_threshold_kpc": float(np.median(d_thr) / units.KPC) if rec.any() else 0,
        "z_launch_kpc_median": float(np.median(z_launch)) if launched.any() else 0,
        "log_tail": tail})
    return checks
