"""Global conservation over a run in which light particles are spawned, fly,
recouple and merge.

The star-forming layer of light/spawn_rate (PRFM, eta_M = 0.5, eta_E = 0.1,
N = 10, rebuild every 0.1 Myr), run for 4 per cent of t_dep (about 11 Myr),
with a short flight (d_rec = 0.1 kpc) and a fast clock (R_200 = 1 kpc,
t_0 = 0.5 Myr: flagged light for 1.4 Myr after recoupling, then
derefined over t_0 ln 1e4 = 4.6 Myr), so that every stage happens inside
the run. No gravity (constant zero field), no cooling. SPHENIX.

Checks:
  1. gas + stars mass at the end against the start, to 1e-6 (float masses);
  2. spawned, recoupled and merged all non-zero;
  3. momentum: the change of sum m v over gas and stars equals the sum of the
     launch kicks m Delta v (log), within 1e-3 of sum m |Delta v|. The
     launches add momentum along +-z; star formation, merging and the hydro
     forces conserve it, the last only to the accuracy of individual
     time-steps (measured 2026-10-20: 1.5e-4, with 6188 spawns, 6015
     recouplings and 2685 merges);
  4. the reservoir identity of spawn_rate, to 1e-9.
"""

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..wind import _common as wind_common
from . import _common
from . import spawn_rate as sr

DESCRIPTION = "light wind: mass and momentum conserved through spawn, flight, recoupling, merging"
TIER = "quick"
BUILDS = {"sph": _common.SPH_BUILD}
FLAGS = sr.FLAGS


def run(ctx):
    p = sr.layer_params(ctx, 0.04)
    p["Wind"].update({"recoupling_distance_kpc": 0.1, "recoupling_kernel_factor": 0.0,
                      "recoupling_time_Myr": 1e4, "light_R200_kpc": 1.0,
                      "light_t0_Myr": 0.5, "light_rebuild_interval_Myr": 0.1})
    ctx.info["dt_rebuild_Myr"] = 0.1
    ctx.swift(p, FLAGS, name="light", timeout=1800)


def analyze(ctx):
    checks = []
    snaps = snapshots(ctx.run_dir)
    first, last = snaps[0], snaps[-1]
    tail = _common.light_tail(ctx.path("light.log"))
    wtail = wind_common.wind_tail(ctx.path("light.log"))

    def mass(s):
        return s.read(0, "Masses", cgs=False).sum() + s.read(4, "Masses", cgs=False).sum()

    def momentum(s):
        P = np.zeros(3)
        for t in (0, 4):
            m = s.read(t, "Masses", cgs=False)
            if len(m):
                P += (m[:, None] * s.read(t, "Velocities", cgs=False)).sum(axis=0)
        return P

    checks.append(Check("gas + stars mass, end / start", mass(last) / mass(first), 1.0,
                        kind="rel", tol=1e-6, source="float masses"))
    checks.append(Check("light particles spawned", tail["n_spawn"], 1, kind="min",
                        source="reservoirs"))
    checks.append(Check("merges", tail["n_merge"], 1, kind="min",
                        source="fast clock: R_200 = 1 kpc, t_0 = 0.5 Myr"))
    rec = wtail["n_recouple"] if wtail else 0
    checks.append(Check("recouplings", rec, 1, kind="min", source="d_rec = 0.1 kpc"))
    dP = momentum(last) - momentum(first)
    P_kick = np.array([tail["Px_launch"], tail["Py_launch"], tail["Pz_launch"]])
    m_l = (ctx.info["m_g_Msun"] * units.SOLAR_MASS / units.UNIT_MASS) / sr.N_LIGHT
    e_w = sr.ETA_E * sr.U_STAR / sr.ETA_M / units.UNIT_ENERGY_PER_MASS
    dv = np.sqrt(2.0 * sr.F_KIN * e_w)
    scale = tail["n_spawn"] * m_l * dv
    checks.append(Check("|dP - sum of kicks| / sum m |Delta v|",
                        np.abs(dP - P_kick).max() / scale, 1e-3, kind="max",
                        source="launch kicks are the only external momentum"))
    m_sp = tail["m_spawn"] - tail["m_source"]
    acc = m_sp + tail["m_rounding"] + tail["m_held"] + tail["m_to_stars"]
    checks.append(Check("added = spawned + rounding + held + to stars", acc,
                        tail["m_added"], kind="rel", tol=1e-9, source="spawn_rate check 1"))
    ctx.info.update({"light_tail": tail, "dP": dP.tolist(), "P_kick": P_kick.tolist(),
                     "kick_scale": scale})
    return checks
