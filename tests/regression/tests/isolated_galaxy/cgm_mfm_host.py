"""Milky Way-mass disc in a hot CGM with GIZMO-MFM hydrodynamics, density
recoupling with merging into a host, and the warm channel with K20a launch speeds
(long tier).

The set-up of isolated_galaxy/two_phase_cgm (H12 disc in a hot CGM,
<data>/ics/<RES>_cgm_disk.hdf5, RES = H12_m1.6e6 by default; SPH and Stars h_max =
150 kpc, 8 top-level cells). An MFM disc needs the CGM: in vacuum, outflow particles
reach h_max with nearly empty kernels and gain energy without bound
(papers-part12-hopkins.md Sec. 5.4). Hydro scheme SWIFT_REG_ARK_HYDRO (default
gizmo-mfm, HLLC). Runs (SWIFT_REG_CGMM_RUNS, default all): nowind and, for the hot
loadings tig, highspec and lowspec of arkenstone_mfm_host, the 'full' variant there
(hot: density recoupling below n_H = 0.013 cm^-3 or 100 Myr, merged into a host;
warm: K20a speeds with v_climb = 62 km/s, the climb to 1 kpc at R = 6 kpc in this
galaxy (combined_cgm), and the warm loading through TIGRESS++ mhd at 1 kpc). Host
runs split particles above 1.5 times the median IC gas mass. Length
SWIFT_REG_CGMM_TEND_MYR (default 1000), snapshots every 10 Myr. SWIFT_REG_SEED as in
arkenstone_mfm_host.

Checks: the checks of arkenstone_mfm_host (end time, realised loadings, merges,
K20a energy, gas plus stellar mass to 2e-2).
"""

import os
import shutil

import h5py
import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import two_phase_cgm as cgm2
from . import arkenstone_twophase as ark2
from . import arkenstone_mfm_host as arkm

DESCRIPTION = "H12 disc in a hot CGM, GIZMO-MFM: density recoupling with host merge plus warm K20a"
TIER = "long"
_build = list(cgm2.BUILDS["wind"])
if arkm.HYDRO != "sphenix":
    _build += [f"--with-hydro={arkm.HYDRO}", "--with-riemann-solver=hllc"]
if arkm.SEED:
    _build += [f"--with-random-seed={arkm.SEED}"]
BUILDS = {"wind": _build}
T_END_MYR = float(os.environ.get("SWIFT_REG_CGMM_TEND_MYR", "1000"))
V_CLIMB = 62.0  # km/s

WINDS = {"nowind": {"loading_model": "none"}}
for _l in ("tig", "highspec", "lowspec"):
    WINDS[f"{_l}_full"] = dict(arkm.WINDS[f"{_l}_full"], warm_climb_speed_kms=V_CLIMB)
_only = os.environ.get("SWIFT_REG_CGMM_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    path = os.path.join(ctx.data_dir, "ics", cgm2.IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: make it with swift-ism-model add_cgm.py")
    with h5py.File(path, "r") as f:  # IC masses in 1e10 Msun
        m_g_msun = float(np.median(f["PartType0/Masses"][:])) * 1e10
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(path, ctx.path(name, cgm2.IC_NAME))
        cgm2.T_END_MYR = T_END_MYR  # read by disc_params
        p = cgm2.disc_params(ctx)
        p["Wind"] = dict(WINDS[name], launch_direction=[0.0, 0.0, 1.0],
                         recoupling_distance_kpc=2.0, recoupling_kernel_factor=2.0,
                         max_timestep_Myr=1.0)
        if "recoupling_time_Myr" not in WINDS[name]:
            p["Wind"]["recoupling_time_Myr"] = 25.0
        arkm.host_params(p, m_g_msun)
        ctx.swift(p, cgm2.FLAGS, subdir=name, name=name, timeout=24 * 3600)


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        first, last = snaps[0], snaps[-1]
        checks.append(Check(f"{name}: end time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-3,
                            source="SWIFT_REG_CGMM_TEND_MYR"))
        tail = None
        if name != "nowind":
            tail = ark2.wind_checks(ctx, name, WINDS[name], last, checks)
            arkm.host_checks(ctx, name, WINDS[name], checks)
        mratio = arkm.mass_check(name, first, last, checks)
        ctx.info[name] = {"wind_phases": tail, "seed": arkm.SEED, "hydro": arkm.HYDRO,
                          "mass_ratio": mratio}
    return checks
