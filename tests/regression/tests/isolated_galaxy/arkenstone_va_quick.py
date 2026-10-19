"""Arkenstone I isolated galaxy, quick variant: the set-up of arkenstone_va at the
coarsest resolution (particle mass 1.3e7 Msun) for 50 Myr with the HighSpec wind.

ICs <data>/ics/ARK_m1.3e7.hdf5 from swift-ism-model tests/arkenstone/make_ics.py.
SWIFT_REG_ARK_RES selects another IC file of arkenstone_va.

Checks:
  1. the run reached 50 Myr;
  2. total mass of gas and stars at the end equals the initial total to 1e-6: star
     formation and wind launching move mass between particle types without creating
     or destroying it, and no particle reaches the box edge (1200 kpc from the centre;
     the CGM ends at 600 kpc and moves at < 100 km/s, 5 kpc in 50 Myr). The tolerance
     is float32 rounding of particle masses summed over ~1e4 particles;
  3. the CGM inflow at the last snapshot, averaged over the thick shell 10-50 kpc
     (sum of m v_r over inflowing gas divided by 40 kpc), equals the analytic cooling
     flow's Mdot = 0.95 Msun/yr (the steady solution the ICs sample) within a factor 2.
     At 1.3e7 Msun the shell holds ~100 particles (Poisson ~10 per cent); the rotation
     added to the ICs, the CGM's self-gravity (left out of the solution, as in S24a)
     and the disc's own disturbance change the flow by tens of per cent in 50 Myr
     (S24a Sec. 3.5; Stern et al. 2023 Sec. 3.1). The check catches a sign or unit
     error in the IC velocities or a potential that does not hold the CGM.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import arkenstone_va as va

DESCRIPTION = "Arkenstone I isolated galaxy, 50 Myr at 1.3e7 Msun with the HighSpec wind"
TIER = "quick"
BUILDS = va.BUILDS
RES = os.environ.get("SWIFT_REG_ARK_RES", "m1.3e7")
IC_NAME = f"ARK_{RES}.hdf5"
T_END_MYR = 50.0
MDOT_CF = 0.95  # Msun/yr, swift_ism_model.arkenstone.cooling_flow() for S24a Table 1


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    path = os.path.join(ctx.data_dir, "ics", IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}: make it with swift-ism-model tests/arkenstone/make_ics.py")
    shutil.copy(path, ctx.path(IC_NAME))
    p = va.parameters(ctx.data_dir, T_END_MYR, 10.0, va.WINDS["highspec"])
    p["InitialConditions"]["file_name"] = IC_NAME
    soft = {"m2e5": 0.2, "m1.6e6": 0.4, "m1.3e7": 0.8}[RES]
    p["Gravity"].update({"max_physical_DM_softening": soft,
                         "max_physical_baryon_softening": soft})
    p["NFWDKPotential"]["epsilon"] = soft
    if p["SPH"].get("particle_splitting"):
        m_g = {"m2e5": 2e5, "m1.6e6": 1.6e6, "m1.3e7": 1.3e7}[RES] / 1e10
        p["SPH"]["particle_splitting_mass_threshold"] = 1.5 * m_g
    ctx.swift(p, va.FLAGS, name="quick", timeout=3600)


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    first, last = snaps[0], snaps[-1]
    checks = [Check("final time [Myr]", last.time_cgs / units.MYR, T_END_MYR, kind="rel",
                    tol=1e-2, source="run completed")]

    def total(s):
        return s.read(0, "Masses").sum() + s.read(4, "Masses").sum()

    checks.append(Check("total mass, end / start", total(last) / total(first), 1.0,
                        kind="rel", tol=1e-6, source="mass moves between types only"))
    x = last.read(0, "Coordinates") - np.full(3, 0.5 * va.BOX_KPC * units.KPC)
    v = last.read(0, "Velocities")
    m = last.read(0, "Masses")
    r = np.linalg.norm(x, axis=1)
    vr = np.sum(x * v, axis=1) / np.maximum(r, 1e-30)
    s = (r > 10 * units.KPC) & (r < 50 * units.KPC) & (vr < 0)
    m_in = -np.sum(m[s] * vr[s]) / (40 * units.KPC) * units.YEAR / units.SOLAR_MASS
    for bound, kind in ((0.5, "min"), (2.0, "max")):
        checks.append(Check(f"inflow 10-50 kpc / cooling-flow Mdot ({kind} {bound})",
                            m_in / MDOT_CF, bound, kind=kind,
                            source="analytic Mdot of the IC flow; factor 2 (docstring)"))
    ctx.info.update({"inflow_0.2r200_Msun_yr": m_in,
                     "stars_formed": int((last.read(4, "BirthTimes") > 0).sum())})
    return checks
