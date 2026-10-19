"""Two-phase wind and CR-carrying wind particles in the Arkenstone I isolated galaxy
(Experiment B, wind-design.md Sec. 10; long tier).

The galaxy, ISM model and diagnostics of isolated_galaxy/arkenstone_va (ICs
<data>/ics/ARK_<res>.hdf5, SWIFT_REG_ARK_RES = m2e5 by default); only the wind
differs. Runs (SWIFT_REG_ARK2_RUNS, comma-separated; default all):
  tig_hot         TIGRESS K20b hot loadings at Sigma_SFR (enthalpy convention)
  tig_2ph         tig_hot plus the warm channel (K20 set: eta_M,warm through K26
                  mhd 1.3 at 2 kpc)
  tig_2ph_v150, tig_2ph_v200  the same with warm launch speeds 150, 200 km/s
  tig_hot_cr     loading_set TIGRESSpp without the warm channel: hot x (0.515,
                  0.40), CR eta_E = 0.3 carried by the hot phase, thermal deposit
  tig_2ph_cr      TIGRESSpp with the warm channel (eta_M,warm through K26 crmhd
                  2.8 at 2 kpc), CR thermal deposit
  tig_2ph_crnone  tig_2ph_cr with cr_deposit none
  lowspec_warm    S24a LowSpec through the warm channel: eta_M = 6.41 at 378 km/s
                  and 3.8e5 K (c_s = 94 km/s), apex / 1.5 kpc recoupling; no hot
  highspec_warm   S24a HighSpec hot (0.32, 0.9, Mach 1 internal) plus the K20 warm
                  channel
Warm channel defaults: 100 km/s, 1e4 K, recoupling at max(1.5 kpc, 2 kernel
supports), 25 Myr or the apex. Length SWIFT_REG_ARK_TEND_MYR (default 1000),
snapshots every 10 Myr. The references nowind, lowspec and highspec are the runs of
arkenstone_va.

Checks (each run):
  1. the run reached its end time;
  2. TIGRESS runs: for each launched phase, the sum of m / eta_M over the launches
     (log 'Wind phases', eta_M at the launch Sigma_SFR) against the stellar mass
     formed, within 3 sigma, sigma^2 = (1 + 1/eta) N_* m_g^2 (each launch stands
     for m / eta of stars in expectation): the loadings realised equal the input
     fits whatever the spread of Sigma_SFR;
  3. CR runs: CR energy launched = released + carried by the decoupled particles
     of the last snapshot (1e-6; the log's sums are doubles, the snapshot is
     written slightly after the last statistics line), deposited = released
     (thermal) or 0 (none).
Recorded: the time series of arkenstone_va.series (SFR, shell fluxes) and the
'Wind phases' totals. Phase-resolved fluxes, the fountain and CGM profiles are
reduced by swift-ism-model tests/twophase/reduce.py.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import arkenstone_va as va
from ..wind import _twophase as tp

DESCRIPTION = "Arkenstone I galaxy with the two-phase wind and CR-carrying wind particles"
TIER = "long"
BUILDS = va.BUILDS
T_END_MYR = va.T_END_MYR

S24A_HIGH = {"loading_model": "constant", "eta_M": 0.32, "eta_E": 0.900,
             "kinetic_fraction": 0.321 / 0.900}
WINDS = {
    "tig_hot": {"loading_model": "TIGRESS"},
    "tig_2ph": {"loading_model": "TIGRESS", "warm_phase": 1},
    "tig_2ph_v150": {"loading_model": "TIGRESS", "warm_phase": 1,
                     "warm_launch_speed_kms": 150.0},
    "tig_2ph_v200": {"loading_model": "TIGRESS", "warm_phase": 1,
                     "warm_launch_speed_kms": 200.0},
    "tig_hot_cr": {"loading_model": "TIGRESS", "loading_set": "TIGRESSpp"},
    "tig_2ph_cr": {"loading_model": "TIGRESS", "loading_set": "TIGRESSpp",
                   "warm_phase": 1},
    "tig_2ph_crnone": {"loading_model": "TIGRESS", "loading_set": "TIGRESSpp",
                       "warm_phase": 1, "cr_deposit": "none"},
    "lowspec_warm": {"loading_model": "constant", "eta_M": 0.0, "eta_E": 0.0,
                     "warm_phase": 1, "warm_eta_M": 6.41,
                     "warm_launch_speed_kms": 378.0, "warm_temperature_K": 3.8e5},
    "highspec_warm": dict(S24A_HIGH, warm_phase=1),
}
_only = os.environ.get("SWIFT_REG_ARK2_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    ics = va.ic_file(ctx.data_dir)
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(ics, ctx.path(name, va.IC_NAME))
        p = va.parameters(ctx.data_dir, T_END_MYR, va.DT_SNAP_MYR, WINDS[name])
        ctx.swift(p, va.FLAGS, subdir=name, name=name, timeout=48 * 3600)


def wind_checks(ctx, name, w, last, checks):
    """Checks 2 and 3 of a run; shared with isolated_galaxy/two_phase_cgm."""
    tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
    if tail is None:
        checks.append(Check(f"{name}: Wind phases line in the log", 0, 1, kind="equal",
                            source="wind_print_stats"))
        return None
    birth = last.read(4, "BirthTimes")
    m0 = last.read(4, "InitialMasses", cgs=False)
    new = birth > 0
    n_star, m_star = int(new.sum()), float(m0[new].sum())
    m_g = float(np.median(m0[new])) if n_star else 1.0
    if w.get("loading_model") == "TIGRESS":
        launched = last.read(0, "WindLaunchCounts", cgs=False) > 0
        phase = last.read(0, "WindPhases", cgs=False)
        eta = last.read(0, "WindLaunchMassLoadings", cgs=False)
        phases = (("hot", 1, "n_hot"),) + ((("warm", 2, "n_warm"),)
                                          if w.get("warm_phase") else ())
        for ph, code, nkey in phases:
            sel = launched & (phase == code)
            e_med = float(np.median(eta[sel])) if sel.any() else 1.0
            checks.append(Check(
                f"{name}: sum m/eta ({ph}) / M_*", tail[f"m_over_eta_{ph}"] / m_star, 1.0,
                kind="sigma", tol=3.0,
                sigma=np.sqrt((1 + 1 / e_med) * max(n_star, 1)) * m_g / m_star,
                source=f"{int(tail[nkey])} launches, N_* = {n_star}, median eta {e_med:.3g}"))
    if tail["E_cr_launch"] > 0:
        carried = float(last.read(0, "WindCREnergies", cgs=False).sum())
        checks.append(Check(f"{name}: CR (released + carried) / launched",
                            (tail["E_cr_release"] + carried) / tail["E_cr_launch"], 1.0,
                            kind="rel", tol=1e-6,
                            source="per-particle doubles; snapshot at the last statistics"))
        dep = tail["E_cr_release"] if w.get("cr_deposit", "thermal") == "thermal" else 0.0
        checks.append(Check(f"{name}: CR deposited (log)", tail["E_cr_deposit"], dep,
                            kind="rel" if dep else "abs", tol=1e-12 if dep else 0.0,
                            source=f"cr_deposit: {w.get('cr_deposit', 'thermal')}"))
    return tail


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        t_end = last.time_cgs / units.MYR
        checks.append(Check(f"{name}: end time [Myr]", t_end, T_END_MYR, kind="rel",
                            tol=1e-3, source="SWIFT_REG_ARK_TEND_MYR"))
        tail = wind_checks(ctx, name, WINDS[name], last, checks)
        res, _ = va.series(ctx.path(name))
        ctx.info[name] = {"wind_phases": tail,
                          **{k: np.asarray(v).tolist() for k, v in res.items()}}
    return checks
