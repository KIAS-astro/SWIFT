"""Milky Way-mass disc in a hot CGM with the combined decoupled wind (wind-design.md
Sec. 11; long tier).

The set-up of isolated_galaxy/two_phase_cgm (H12 disc in a hot CGM,
<data>/ics/<RES>_cgm_disk.hdf5, RES = H12_m1.6e6 by default; SPH and Stars h_max =
150 kpc, 8 top-level cells) with the recoupling of isolated_galaxy/arkenstone_combined
(hot at max(10 kpc, 2 kernel supports) or 50 Myr; warm at a height of max(1.5 kpc, 2
kernel supports), 50 Myr or the apex). Runs (SWIFT_REG_CGMC_RUNS, default all):
  nowind, tig_hot10, tig_w100, tig_wk20a, tig_wv60, tig_wk20a1k
with TIGRESS K20b hot loadings and the K20 warm set; tig_wk20a, tig_wv60 and
tig_wk20a1k draw warm speeds from the K20a PDF as in arkenstone_combined, with
v_climb = 62 km/s, the climb to 1 kpc at R = 6 kpc (the median radius of the IC gas
within 0.5 kpc of the midplane; direct summation over the IC particles plus the
Hernquist halo gives Phi(1 kpc) - Phi(0) = 1950 (km/s)^2 there,
3030 at R = 4 kpc and 1225 at 8 kpc). Length SWIFT_REG_CGMC_TEND_MYR (default 1000),
snapshots every 10 Myr. SWIFT_REG_SEED as in arkenstone_combined.

Checks: end time; the loading checks of arkenstone_twophase; the recoupling-trigger
and K20a PDF checks of arkenstone_combined.
"""

import os
import shutil

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import two_phase_cgm as cgm2
from . import arkenstone_twophase as ark2
from . import arkenstone_combined as arkc

DESCRIPTION = "H12 disc in a hot CGM with the combined wind (hot at 10 kpc plus warm)"
TIER = "long"
BUILDS = {"wind": cgm2.BUILDS["wind"]
          + ([f"--with-random-seed={arkc.SEED}"] if arkc.SEED else [])}
T_END_MYR = float(os.environ.get("SWIFT_REG_CGMC_TEND_MYR", "1000"))
V_CLIMB = 62.0  # km/s

WINDS = {
    "nowind": {"loading_model": "none"},
    "tig_hot10": arkc.WINDS["tig_hot10"],
    "tig_w100": arkc.WINDS["tig_w100"],
    "tig_wk20a": dict(arkc.WINDS["tig_wk20a"], warm_climb_speed_kms=V_CLIMB),
    "tig_wv60": dict(arkc.WINDS["tig_wv60"], warm_climb_speed_kms=V_CLIMB),
    "tig_wk20a1k": dict(arkc.WINDS["tig_wk20a1k"], warm_climb_speed_kms=V_CLIMB),
}
_only = os.environ.get("SWIFT_REG_CGMC_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    path = os.path.join(ctx.data_dir, "ics", cgm2.IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: make it with swift-ism-model add_cgm.py")
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(path, ctx.path(name, cgm2.IC_NAME))
        cgm2.T_END_MYR = T_END_MYR  # read by disc_params
        p = cgm2.disc_params(ctx)
        p["Wind"] = dict(WINDS[name], launch_direction=[0.0, 0.0, 1.0],
                         **arkc.recoupling())
        ctx.swift(p, cgm2.FLAGS, subdir=name, name=name, timeout=24 * 3600)


def analyze(ctx):
    checks = []
    for name in RUNS:
        last = snapshots(ctx.path(name))[-1]
        checks.append(Check(f"{name}: end time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-3,
                            source="SWIFT_REG_CGMC_TEND_MYR"))
        if name != "nowind":
            tail = ark2.wind_checks(ctx, name, WINDS[name], last, checks)
            arkc.combined_checks(ctx, name, WINDS[name], last, checks, V_CLIMB)
            ctx.info[name] = {"wind_phases": tail, "seed": arkc.SEED}
    return checks
