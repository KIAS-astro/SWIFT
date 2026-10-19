"""Isolated disc in a hot CGM with PRFM-int and the TIGRESS wind: whole
particles against wind carriers (X3, long tier).

ICs: the H12 disc with a hot CGM (swift-ism-model add_cgm.py),
<data>/ics/<RES>_cgm_disk.hdf5, RES = H12_m1.3e7 (default; gas particle mass
1.3e7 Msun, softening 1 kpc) or H12_m1.6e6 (SWIFT_REG_ISOGAL_RES). The gas
carries its own metals (EAGLEChemistry:init_abundance_metal = -1). Halo
particles at 30-170 kpc need h = 20-60 kpc, so SPH:h_max = 150 kpc (it must
stay below the top-level cell, 4565 kpc / 16 = 285 kpc). Otherwise the
set-up of isolated_galaxy/prfm_wind: Hernquist halo, PRFM-int with OK22,
TIGRESS effective EoS floor, EAGLE cooling, K20b hot-phase loadings at
Sigma_SFR = W / Upsilon, launch along +-z.

Recoupling at d_rec = 10 kpc (Sullivan et al. 2026 fiducial; at least
5 kpc needed against the inward creep of density recoupling), or after
t_rec = 100 Myr (Bennett et al. 2025), with d_rec at least 2 kernel
supports at launch. Runs (SWIFT_REG_CARRIER_RUNS, default all):
  particle     whole-particle decoupled wind, d_rec = 10 kpc
  carrier8     carriers, N_c = 8, d_rec = 10 kpc
  carrier8_d5  carriers, N_c = 8, d_rec = 5 kpc
Length SWIFT_REG_CARRIER_TEND_MYR (default 300), snapshots every 10 Myr.

Checks:
  1. each run reached its end time;
  2. winds launched in each run;
  3. carriers: mass bookkeeping, launched = deposited + still flying (the
     carriers in the last snapshot), relative 1e-5 (float masses summed in
     double);
  4. carriers: every deposit by distance has d >= d_rec and every deposit
     by time has t - t_launch >= t_rec (wind_carriers.txt);
  5. carriers: energy added by the internal-energy floor below 1e-3 of the
     deposited energy.
Recorded (ctx.info): SFR in 10 Myr bins (stars only, carriers excluded by
WindCarrier), launched and deposited/recoupled mass and energy, fraction
deposited by time, radius, |z| and distance at deposit/recoupling, and the
gas particle masses at the end over the initial m_g (disc gas, |z| < 2 kpc
and R < 30 kpc, and halo gas: min, max, percentiles, numbers below 0.5 m_g
and above 2 m_g), since carriers move mass between particles.
"""

import os
import shutil

import h5py
import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import prfm_disc
from ..wind import _common as wind_common

DESCRIPTION = "isolated disc in a hot CGM: whole-particle wind vs wind carriers (N_c = 8)"
TIER = "long"
BUILDS = {"wind": prfm_disc.BUILDS["prfm"] + ["--with-wind=TIGRESS"]}

RES = os.environ.get("SWIFT_REG_ISOGAL_RES", "H12_m1.3e7")
SOFTENING = {"H12_m1.6e6": 0.5, "H12_m1.3e7": 1.0}[RES]
IC_NAME = f"{RES}_cgm_disk.hdf5"
MYR = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_CARRIER_TEND_MYR", "300"))
T_REC_MYR = 100.0
H_MAX = 150.0  # kpc

ALL_RUNS = {
    "particle": {"mode": "particle", "recoupling_distance_kpc": 10.0},
    "carrier8": {"mode": "carrier", "carriers_per_particle": 8.0,
                 "recoupling_distance_kpc": 10.0},
    "carrier8_d5": {"mode": "carrier", "carriers_per_particle": 8.0,
                    "recoupling_distance_kpc": 5.0},
}
_only = os.environ.get("SWIFT_REG_CARRIER_RUNS")
RUNS = {k: v for k, v in ALL_RUNS.items() if not _only or k in _only.split(",")}
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
    p["Scheduler"] = {"max_top_level_cells": 16, "tasks_per_cell": 200}
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
    for name, w in RUNS.items():
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(path, ctx.path(name, IC_NAME))
        p = disc_params(ctx)
        p["Wind"] = dict({"loading_model": "TIGRESS", "launch_direction": [0.0, 0.0, 1.0],
                          "recoupling_kernel_factor": 2.0,
                          "recoupling_time_Myr": T_REC_MYR, "max_timestep_Myr": 1.0}, **w)
        flags = FLAGS + (["--feedback"] if w["mode"] == "carrier" else [])
        ctx.swift(p, flags, subdir=name, name=name, timeout=12 * 3600)


def sfr_history(last):
    birth = last.read(4, "BirthTimes") / units.MYR
    m_init = last.read(4, "InitialMasses")
    star = birth > 0
    if last.has(4, "WindCarrier"):
        star &= last.read(4, "WindCarrier", cgs=False) == 0
    edges = np.arange(0, T_END_MYR + 1e-6, 10.0)
    sfr = np.histogram(birth[star], edges, weights=m_init[star])[0] / (
        units.SOLAR_MASS * 10e6)
    return sfr, float(m_init[star].sum() / units.SOLAR_MASS)


def gas_mass_distribution(first, last, centre):
    """Gas particle masses at the end over the initial median m_g, for disc
    gas (|z| < 2 kpc, R < 30 kpc) and halo gas: min, max, 1/16/50/84/99th
    percentiles, and the numbers below 0.5 m_g and above 2 m_g."""
    m_g = np.median(first.read(0, "Masses", cgs=False))
    m = last.read(0, "Masses", cgs=False) / m_g
    x = last.read(0, "Coordinates", cgs=False) - centre
    disc = (np.abs(x[:, 2]) < 2.0) & (np.hypot(x[:, 0], x[:, 1]) < 30.0)
    out = {}
    for name, sel in (("disc", disc), ("halo", ~disc)):
        q = m[sel]
        if q.size == 0:
            out[name] = None
            continue
        out[name] = {"N": int(q.size), "min": float(q.min()), "max": float(q.max()),
                     "p1_16_50_84_99": np.percentile(q, [1, 16, 50, 84, 99]).tolist(),
                     "N_below_0.5": int((q < 0.5).sum()), "N_above_2": int((q > 2).sum())}
    return out


def analyze(ctx):
    checks = []
    E_unit = units.UNIT_MASS * units.UNIT_VELOCITY ** 2
    for name, w in RUNS.items():
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        with h5py.File(last.path) as f:
            centre = 0.5 * np.ravel(f["Header"].attrs["BoxSize"])[:3]
        sfr, m_star = sfr_history(last)
        info = {"sfr_Msun_yr": sfr.tolist(), "M_star_formed_Msun": m_star,
                "sfr_mean_last100_Msun_yr": float(sfr[-10:].mean()),
                "gas_mass_over_m_g": gas_mass_distribution(snaps[0], last, centre)}
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-2, source="run completed"))

        if w["mode"] == "particle":
            counts = last.read(0, "WindLaunchCounts", cgs=False)
            t_rec = last.read(0, "WindRecouplingTimes", cgs=False)
            rec = t_rec > 0
            tail = wind_common.wind_tail(ctx.path(name, f"{name}.log"))
            checks.append(Check(f"{name}: launches", int(counts.sum()), 1, kind="min",
                                source="PRFM SF with the TIGRESS wind"))
            x = (last.read(0, "WindLaunchPositions", cgs=False)
                 + last.read(0, "WindDisplacements", cgs=False))[rec] - centre
            t_l = last.read(0, "WindLaunchTimes", cgs=False)[rec]
            by_time = (t_rec[rec] - t_l) >= T_REC_MYR * MYR * (1 - 1e-6)
            info.update({
                "launched_N": tail["n_launch"] if tail else None,
                "launched_M_Msun": tail["m_launch"] * 1e10 if tail else None,
                "launched_E_erg": (tail["E_kin_launch"] + tail["E_th_launch"]) * E_unit
                if tail else None,
                "recoupled_N": tail["n_recouple"] if tail else None,
                "recoupled_M_Msun": tail["m_recouple"] * 1e10 if tail else None,
                "recoupled_E_erg": (tail["E_kin_recouple"] + tail["E_th_recouple"]) * E_unit
                if tail else None,
                "frac_by_time": float(by_time.mean()) if rec.any() else None,
                "r_rec_kpc_quartiles": np.percentile(np.linalg.norm(x, axis=1),
                                                     [25, 50, 75]).tolist()
                if rec.any() else None,
                "absz_rec_kpc_median": float(np.median(np.abs(x[:, 2]))) if rec.any()
                else None})
        else:
            tail = wind_common.carrier_tail(ctx.path(name, f"{name}.log"))
            ev = wind_common.carrier_events(ctx.path(name, "wind_carriers.txt"))
            checks.append(Check(f"{name}: carriers launched", tail["n_launch"] if tail else 0,
                                1, kind="min", source="PRFM SF with carriers"))
            flying = 0.0
            if last.npart[4] > 0 and last.has(4, "WindCarrier"):
                fl = last.read(4, "WindCarrier", cgs=False) > 0
                flying = float(last.read(4, "Masses", cgs=False)[fl].sum())
            if tail:
                checks.append(Check(f"{name}: (deposited + flying) / launched mass",
                                    (tail["m_deposit"] + flying) / tail["m_launch"], 1.0,
                                    kind="rel", tol=1e-5, source="log totals + last snapshot"))
            crit = np.array([e["criterion"] for e in ev])
            d = np.array([e["distance"] for e in ev])
            d_thr = np.array([e["d_rec"] for e in ev])
            age = np.array([e["t_deposit"] - e["t_launch"] for e in ev])
            ok = np.where(crit == 1, d >= d_thr * (1 - 1e-6),
                          age >= T_REC_MYR * MYR * (1 - 1e-6))
            checks.append(Check(f"{name}: deposits meeting their criterion (fraction)",
                                float(ok.mean()) if len(ev) else 1.0, 1.0, kind="equal",
                                source="wind_carriers.txt"))
            if tail:
                E_dep = tail["E_kin_deposit"] + tail["E_th_deposit"]
                checks.append(Check(f"{name}: floor energy / deposited energy",
                                    tail["E_floor"] / E_dep if E_dep > 0 else 0.0, 1e-3,
                                    kind="max", source="u floor rarely applies"))
            x = np.array([e["x_deposit"] for e in ev]) - centre if ev else np.zeros((1, 3))
            info.update({
                "launched_N": tail["n_launch"] if tail else None,
                "launched_M_Msun": tail["m_launch"] * 1e10 if tail else None,
                "launched_E_erg": (tail["E_kin_launch"] + tail["E_th_launch"]) * E_unit
                if tail else None,
                "launch_net_E_erg": tail["E_net_launch"] * E_unit if tail else None,
                "deposited_N": tail["n_deposit"] if tail else None,
                "deposited_M_Msun": tail["m_deposit"] * 1e10 if tail else None,
                "deposited_E_erg": (tail["E_kin_deposit"] + tail["E_th_deposit"]) * E_unit
                if tail else None,
                "deposit_E_pot_erg": tail["E_pot_deposit"] * E_unit if tail else None,
                "floor_E_erg": tail["E_floor"] * E_unit if tail else None,
                "floor_E_over_deposited": tail["E_floor"] / max(
                    tail["E_kin_deposit"] + tail["E_th_deposit"], 1e-300) if tail else None,
                "flying_M_Msun": flying * 1e10, "retries": tail["n_retry"] if tail else None,
                "frac_by_time": float((crit == 2).mean()) if len(ev) else None,
                "r_dep_kpc_quartiles": np.percentile(np.linalg.norm(x, axis=1),
                                                     [25, 50, 75]).tolist(),
                "absz_dep_kpc_median": float(np.median(np.abs(x[:, 2]))),
                "d_dep_kpc_median": float(np.median(d)) if len(ev) else None,
                "ngb_median": float(np.median([e["n_ngb"] for e in ev])) if ev else None,
                "h_dep_kpc_median": float(np.median([e["h"] for e in ev])) if ev else None})
        ctx.info[name] = info
    return checks
