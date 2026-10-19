"""Experiment A in isolated galaxies (long tier): distance recoupling against
pressure-balance recoupling and loadings at the recoupling height
(ai-notes wind-design.md Sec. 9).

SWIFT_REG_WR_GALAXY selects the galaxy:
  ark  the Arkenstone I galaxy of isolated_galaxy/arkenstone_va (S24a; NFW
       1e11 Msun with the DK14 outer term, rotating cooling-flow CGM, 2.4 Mpc
       non-periodic box, SPH h_max 120 kpc), resolution SWIFT_REG_ARK_RES
       (m2e5 default), S24a's HighSpec or LowSpec constant loadings
       (internal-energy convention, as in V-A);
  h12  the H12 disc in the hot CGM (<data>/ics/<RES>_cgm_disk.hdf5,
       RES = SWIFT_REG_ISOGAL_RES, H12_m1.6e6 default), Hernquist halo,
       PRFM-int OK22, TIGRESS loadings (enthalpy convention), metals from
       the ICs, SPH and Stars h_max 150 kpc, 8 top-level cells (box 4565 kpc,
       top-level cell 571 kpc).
Runs (SWIFT_REG_WR_RUNS, comma-separated; for ark each name is prefixed by
the loading set, e.g. highspec_pres3):
  nowind  no launches
  dist    distance recoupling, d_rec = max(2 kpc, 2 kernel supports),
          t_rec = 25 Myr (WD5; the V-A set-up)
  pres05  pressure recoupling, CC85 R = 0.5 kpc (the PRFM cylinder),
          d_min = max(1 kpc, 2 kernel supports), backstops 50 kpc, 25 Myr
  pres3   the same with R = 3 kpc
  zrec    distance recoupling as dist, loadings times eta(d_rec)/eta(H)
  pres3dt pres3 with the decoupled time-step cap 0.1 Myr instead of 1 Myr (how
          much of the recoupling distance is overshoot within one step)
Length SWIFT_REG_WR_TEND_MYR (default 1000), snapshots every 10 Myr.
SWIFT_REG_WR_SEED (optional) builds with --with-random-seed=<seed>, for repeats
that differ only in the random draws.

Checks:
  1. each run reached its end time;
  2. pressure runs: every flight that ended by pressure (reason 4) has
     d >= d_min and recorded P_w <= P_amb (float storage, 1e-6);
  3. pressure runs: no flight ended by pressure before d_min, and none
     ended by distance short of d_rec;
  4. zrec runs: the recorded eta_M and eta_E at launch over the input
     loadings equal the K20b factors at the recorded Sigma_SFR and z = d_rec
     (clamped at 1 kpc), relative 1e-5.
Recorded: launches, flights by reason, recoupling distance percentiles.
The S24a diagnostics are computed from the kept snapshots by swift-ism-model
tests/wind_recoupling/reduce.py.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import arkenstone_va as ark
from . import prfm_disc

GALAXY = os.environ.get("SWIFT_REG_WR_GALAXY", "ark")
DESCRIPTION = "Experiment A: distance vs pressure recoupling vs loadings at d_rec"
TIER = "long"
if GALAXY == "ark":
    BUILDS = {"ark": list(ark.BUILDS["ark"])}
else:
    BUILDS = {"h12": prfm_disc.BUILDS["prfm"] + ["--with-wind=TIGRESS"]}
# Another realisation of the random draws (launch, direction, star formation):
# SWIFT_REG_WR_SEED is passed to configure as --with-random-seed
if os.environ.get("SWIFT_REG_WR_SEED"):
    for b in BUILDS.values():
        b.append("--with-random-seed=" + os.environ["SWIFT_REG_WR_SEED"])

MYR = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_WR_TEND_MYR", "1000"))
H12_RES = os.environ.get("SWIFT_REG_ISOGAL_RES", "H12_m1.6e6")
H12_SOFT = {"H12_m1.6e6": 0.5, "H12_m1.3e7": 1.0}[H12_RES]
H12_IC = f"{H12_RES}_cgm_disk.hdf5"
H_MAX_H12 = 150.0  # kpc

VARIANTS = {
    "dist": {"recoupling": "distance", "recoupling_distance_kpc": 2.0},
    "pres05": {"recoupling": "pressure", "cc85_launch_radius_kpc": 0.5,
               "recoupling_distance_kpc": 50.0, "recoupling_min_distance_kpc": 1.0},
    "pres3": {"recoupling": "pressure", "cc85_launch_radius_kpc": 3.0,
              "recoupling_distance_kpc": 50.0, "recoupling_min_distance_kpc": 1.0},
    "zrec": {"recoupling": "distance", "recoupling_distance_kpc": 2.0,
             "loading_height": "recoupling"},
    "pres3dt": {"recoupling": "pressure", "cc85_launch_radius_kpc": 3.0,
                "recoupling_distance_kpc": 50.0, "recoupling_min_distance_kpc": 1.0,
                "max_timestep_Myr": 0.1},
}
COMMON = {"recoupling_kernel_factor": 2.0, "recoupling_time_Myr": 25.0,
          "max_timestep_Myr": 1.0, "launch_direction": [0.0, 0.0, 1.0]}


def all_runs():
    runs = {}
    if GALAXY == "ark":
        runs["nowind"] = {"loading_model": "none"}
        for spec in ("highspec", "lowspec"):
            for v, w in VARIANTS.items():
                runs[f"{spec}_{v}"] = {**ark.WINDS[spec], **COMMON, **w}
    else:
        runs["nowind"] = {"loading_model": "none"}
        for v, w in VARIANTS.items():
            runs[v] = {"loading_model": "TIGRESS", **COMMON, **w}
    return runs


_only = os.environ.get("SWIFT_REG_WR_RUNS")
RUNS = {k: v for k, v in all_runs().items() if not _only or k in _only.split(",")}
FLAGS = ["--hydro", "--self-gravity", "--external-gravity", "--stars",
         "--star-formation", "--cooling", "--limiter", "--sync"]


def h12_params(ctx):
    p = params.eagle_base(H12_IC, T_END_MYR * MYR, ctx.data_dir, 10.0 * MYR,
                          dt_max=1e-2, cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
    p["Snapshots"]["compression"] = 4
    p["Gravity"].update({"max_physical_DM_softening": H12_SOFT,
                         "max_physical_baryon_softening": H12_SOFT})
    p["SPH"].update({"h_min_ratio": 0.1, "h_max": H_MAX_H12, "minimal_temperature": 100.0})
    p["Stars"]["h_max"] = H_MAX_H12
    p["EAGLEChemistry"]["init_abundance_metal"] = -1
    p["Scheduler"] = {"max_top_level_cells": 8, "tasks_per_cell": 200}
    p["HernquistPotential"] = {
        "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
        "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
        "bulgefraction": 0.014, "timestep_mult": 0.01, "epsilon": H12_SOFT}
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
    if GALAXY == "ark":
        ics, ic_name = ark.ic_file(ctx.data_dir), ark.IC_NAME
    else:
        ics, ic_name = os.path.join(ctx.data_dir, "ics", H12_IC), H12_IC
        if not os.path.exists(ics):
            raise FileNotFoundError(f"{ics}: make it with swift-ism-model add_cgm.py")
    for name, w in RUNS.items():
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(ics, ctx.path(name, ic_name))
        if GALAXY == "ark":
            p = ark.parameters(ctx.data_dir, T_END_MYR, 10.0, w)
            p["Wind"] = dict(w)
        else:
            p = h12_params(ctx)
            p["Wind"] = dict(w)
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=48 * 3600)


def fit_factor(z_kpc, sig, q):
    """K20b height factor eta(z)/eta(H) (wind-design.md Sec. 9.1)."""
    fits = {"500": {"M": (-0.984, -0.145), "E": (-0.953, -0.006)},
            "1000": {"M": (-1.185, -0.100), "E": (-1.381, 0.013)},
            "H": {"M": (-0.86, -0.07), "E": (-0.70, 0.14)}}
    ls = np.log10(np.clip(sig, 1e-4, 1.0))
    s = np.clip((np.log10(z_kpc) - np.log10(0.5)) / np.log10(2.0), 0.0, 1.0)
    lo = fits["500"][q][0] + fits["500"][q][1] * ls
    hi = fits["1000"][q][0] + fits["1000"][q][1] * ls
    h = fits["H"][q][0] + fits["H"][q][1] * ls
    return 10.0 ** (lo + s * (hi - lo) - h)


def analyze(ctx):
    checks = []
    for name, w in RUNS.items():
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-2, source="run completed"))
        if w.get("loading_model") == "none":
            continue
        rd = lambda f: last.read(0, f, cgs=False)  # noqa: E731
        counts = rd("WindLaunchCounts")
        done = rd("WindRecouplingTimes") > 0
        reason = rd("WindRecouplingReasons")[done]
        d = rd("WindRecouplingDistances")[done]
        dmin = rd("WindRecouplingMinDistances")[done]
        dthr = rd("WindRecouplingDistanceThresholds")[done]
        info = {"launches": int(counts.sum()), "flights_done_last": int(done.sum()),
                "by_reason": {int(k): int((reason == k).sum()) for k in (1, 3, 4)},
                "d_rec_kpc_p16_50_84": np.percentile(d, [16, 50, 84]).tolist()
                if done.any() else []}
        if w.get("recoupling") == "pressure":
            P_w = rd("WindRecouplingWindPressures")[done]
            P_a = rd("WindRecouplingAmbientPressures")[done]
            p = reason == 4
            checks.append(Check(f"{name}: pressure flights below d_min", int((d[p] < dmin[p]).sum()),
                                0, kind="equal", source="trigger needs d >= d_min"))
            checks.append(Check(f"{name}: pressure flights with P_w > P_amb",
                                int((P_w[p] > P_a[p] * (1 + 1e-6)).sum()), 0, kind="equal",
                                source="trigger P_w <= P_amb"))
            checks.append(Check(f"{name}: distance flights short of d_rec",
                                int((d[reason == 1] < dthr[reason == 1] * (1 - 1e-6)).sum()), 0,
                                kind="equal", source="backstop d >= d_rec"))
            info["P_w_over_P_amb_median"] = float(np.median(P_w[p] / P_a[p])) if p.any() else None
        if w.get("loading_height") == "recoupling":
            lau = counts > 0
            sig = rd("WindLaunchSigmaSFR")[lau]
            eM = rd("WindLaunchMassLoadings")[lau]
            eE = rd("WindLaunchEnergyLoadings")[lau]
            z = rd("WindRecouplingDistanceThresholds")[lau]
            if w["loading_model"] == "constant":
                base_M, base_E = w["eta_M"], w["eta_E"]
            else:
                ls = np.log10(np.clip(sig, 1e-4, 1.0))
                base_M = 10.0 ** (-0.86 - 0.07 * ls)
                base_E = 10.0 ** (-0.70 + 0.14 * ls)
            rM = eM / base_M / fit_factor(z, sig, "M")
            rE = eE / base_E / fit_factor(z, sig, "E")
            checks.append(Check(f"{name}: max |eta_M / (input f_M) - 1|", np.abs(rM - 1).max(),
                                1e-5, kind="max", source="Sec. 9.1 height factor"))
            checks.append(Check(f"{name}: max |eta_E / (input f_E) - 1|", np.abs(rE - 1).max(),
                                1e-5, kind="max", source="Sec. 9.1 height factor"))
        ctx.info[name] = info
    return checks
