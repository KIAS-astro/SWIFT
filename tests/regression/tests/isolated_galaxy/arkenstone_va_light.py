"""The Arkenstone I isolated galaxy (test V-A) with light wind particles (long tier).

The set-up, ICs, diagnostics and wind loadings of arkenstone_va (S24a Table 2,
internal-energy convention), with Wind:mode light: the wind mass is drawn from
per-particle reservoirs into light particles of m_l = m_g / N at tree rebuilds
(forced every 1 Myr), launched as in the whole-particle wind, recoupled after 2 kpc
(or 2 kernel supports, or 25 Myr), and merged back by the refinement clock of
Bennett et al. (2025) with R200 = 97.94 kpc, t_0 = 10 Myr, k = 0.25
(light-particles-design.md Sec. 8). N = SWIFT_REG_ARK_LIGHT_N (default 10, the
mass-ratio cap). Resolution SWIFT_REG_ARK_RES (default m2e5), runs
SWIFT_REG_ARK_RUNS (highspec, lowspec; default both), length
SWIFT_REG_ARK_TEND_MYR (default 1000). The hydro scheme is the build's
(SWIFT_REG_ARK_HYDRO = sphenix or gizmo-mfm).

Checks (each run):
  1. it reached its end time;
  2. reservoir bookkeeping from the log: added = spawned + rounding + held + to
     stars, to 1e-9 (double sums, 10-digit log);
  3. rounding within 3 sigma of zero, sigma = (m_spawned m_l)^1/2 (each draw has
     variance at most x m_l^2 for a mean x m_l);
  4. gas + stars at the end against the initial total, to 1e-6 (float masses;
     merging and spawning move mass between particles only);
  5. highspec: the smoke band of arkenstone_va for eta_E at 0.1, 0.2, 0.5 r200
     (0.3-2.7, time averages over 0.2 Gyr to the end).
Recorded for the comparison with the whole-particle V-A runs: SFR and shell-flux
series, light-particle counts, and the wall-clock time.
"""

import os
import shutil

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..light import _common as light_common
from . import arkenstone_va as va

DESCRIPTION = "Arkenstone I galaxy with light wind particles (m_g / N), HighSpec and LowSpec"
TIER = "long"
HYDRO = os.environ.get("SWIFT_REG_ARK_HYDRO", "sphenix")
_build = list(va.BUILDS["ark"])
if HYDRO != "sphenix":
    _build += [f"--with-hydro={HYDRO}", "--with-riemann-solver=hllc"]
BUILDS = {"ark": _build}
N_LIGHT = float(os.environ.get("SWIFT_REG_ARK_LIGHT_N", "10"))
M_G_MSUN = {"m2e5": 2e5, "m1.6e6": 1.6e6, "m1.3e7": 1.3e7}[va.RES]
_only = os.environ.get("SWIFT_REG_ARK_RUNS")
RUNS = [k for k in ("highspec", "lowspec") if not _only or k in _only.split(",")]


def run(ctx):
    for t in (va.params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    ics = va.ic_file(ctx.data_dir)
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(ics, ctx.path(name, va.IC_NAME))
        p = va.parameters(ctx.data_dir, va.T_END_MYR, va.DT_SNAP_MYR, va.WINDS[name])
        p["Wind"].update(light_common.light_params(
            M_G_MSUN, N_LIGHT, light_R200_kpc=va.R200_KPC, light_rebuild_interval_Myr=1.0))
        ctx.swift(p, va.FLAGS, subdir=name, name=name, timeout=48 * 3600)


def analyze(ctx):
    checks = []
    msun = units.SOLAR_MASS / units.UNIT_MASS
    m_l = M_G_MSUN / N_LIGHT * msun
    for name in RUNS:
        res, last = va.series(ctx.path(name))
        first = snapshots(ctx.path(name))[0]
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                            va.T_END_MYR, kind="rel", tol=1e-2, source="run completed"))
        tail = light_common.light_tail(ctx.path(name, f"{name}.log"))
        if tail is None:
            checks.append(Check(f"{name}: log has a Light totals line", 0, 1, kind="equal"))
            continue
        m_sp = tail["m_spawn"] - tail["m_source"]
        acc = m_sp + tail["m_rounding"] + tail["m_held"] + tail["m_to_stars"]
        checks.append(Check(f"{name}: added = spawned + rounding + held + to stars", acc,
                            tail["m_added"], kind="rel", tol=1e-9,
                            source="reservoir mass moves, never created or lost"))
        sig = np.sqrt(max(m_sp, m_l) * m_l)
        checks.append(Check(f"{name}: rounding / added", tail["m_rounding"] / tail["m_added"],
                            0.0, kind="sigma", tol=3.0, sigma=sig / tail["m_added"],
                            source="stochastic rounding, variance <= m_spawned m_l"))

        def total(s):
            return s.read(0, "Masses").sum() + s.read(4, "Masses").sum()

        checks.append(Check(f"{name}: total mass, end / start", total(last) / total(first),
                            1.0, kind="rel", tol=1e-6, source="mass moves between particles"))
        info = {k: np.asarray(v).tolist() for k, v in res.items()}
        info["light_tail"] = tail
        light = last.read(0, "WindLight", cgs=False) > 0
        info["n_light_end"] = int(light.sum())
        info["n_gas_end"] = int(len(light))
        ctx.info[name] = info
        if name == "highspec":
            t = res["t"]
            w = t >= 200.0
            sfr = res["sfr"][w].mean() if w.any() else np.nan
            for f in va.SHELLS[:3]:
                eta_E = res[f"eout_{f}"][w].mean() / (va.U_STAR * sfr * units.SOLAR_MASS)
                for bound, kind in ((0.3, "min"), (2.7, "max")):
                    checks.append(Check(f"highspec: time-avg eta_E at {f} r200 ({kind} {bound})",
                                        eta_E, bound, kind=kind,
                                        source="arkenstone_va check 2 (smoke band)"))
    return checks
