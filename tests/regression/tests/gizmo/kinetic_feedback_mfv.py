"""GIZMO-MFV: EAGLE kinetic SN feedback must reach the gas momentum.

gizmo/kinetic_feedback built with GIZMO-MFV instead of MFM; same set-up,
expected values and tolerances.

Without GIZMO_FIX_PARTICLES and GIZMO_STEER_MOTION (the default),
MFV's hydro_velocities_set sets v_full = momentum / mass at every kick, as
MFM does, so a feedback kick added to xp->v_full is erased at the next kick
unless hydro_velocities_absorb_external_kick moves it into the momentum.
Measured 2026-10-20 (gizmo-mfv, HLLC, 8 threads): energy gained / E_SN =
0.062 (61 sigma) with an empty absorb function, 1.012 (0.8 sigma) with the
absorb; net momentum fraction 1.8e-4 and 1.5e-5.
"""

from . import kinetic_feedback as _k

DESCRIPTION = "GIZMO-MFV: EAGLE kinetic feedback kicks reach the gas (energy = CCSN budget)"
TIER = "long"
BUILDS = {"kinetic": ["--with-hydro=gizmo-mfv" if f == "--with-hydro=gizmo-mfm" else f
                      for f in _k.BUILDS["kinetic"]]}

run = _k.run
analyze = _k.analyze
