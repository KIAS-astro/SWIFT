"""Host merge under GIZMO-MFM: wind/recouple_density built with --with-hydro=gizmo-mfm.

The slab, the eight hand-decoupled wind particles and the three runs ('self',
'host', 'host_drift_all') of wind/recouple_density. In MFM a merge goes through
hydro_set_mass (scales the conserved momentum and energy with the mass, keeping v
and u), a change of v_full that the host's next kick moves into the conserved
momentum, and the energy setter (E = m u). The 'host_drift_all' run merges into
active hosts whose density-loop sums (wcount, geometry) have just been zeroed; the
merge must leave them alone. Snapshot velocities are v_full drifted to the output
time, so a momentum change that never reached the conserved momentum would undo
itself at the host's next kick and show as a momentum error.

Checks: those of wind/recouple_density except the momentum and energy bounds,
which are replaced by
 10. host runs: |P - P0| at the end, when every particle is synchronised, below
     1e-5 of the wind particles' sum m |v| (measured 1-2e-7); at any snapshot below
     twice the self run's maximum (2.3e-4 measured: snapshot velocities of
     particles in mid-step carry a dt/2 drift);
 11. host runs: the energy history differs from the self run's by at most 5e-5 of
     the total (measured 3.7e-5); the self run's own excursion (8.5e-5, against
     2.9e-5 with SPHENIX; MFM conserves total energy only to O(dp^2 / 2m) per step,
     papers-part12-hopkins.md Sec. 5.1) is recorded with a 1e-2 bound;
 12. host runs: the summed mass of hosts above 1.5 base masses, less one base mass
     each, equals the merged wind mass to 1e-6.
Negative control: with the host's v_full left at its conserved momentum over mass
(the merged velocity never applied), checks 10 (end) and 11 fail in both host runs
(momentum 2.6e-4, small because the eight wind momenta cancel in pairs; energy
1.3e-2).
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common
from . import recouple_density as rd

DESCRIPTION = "host merge under GIZMO-MFM conserves mass, momentum and energy; active hosts"
TIER = "quick"
BUILDS = {"wind_mfm": _common.WIND_BUILD + ["--with-hydro=gizmo-mfm",
                                            "--with-riemann-solver=hllc"]}
FLAGS = rd.FLAGS
TOL_E_MERGE = 5e-5


def run(ctx):
    rd.run(ctx)


def analyze(ctx):
    checks = []
    base = rd.analyze(ctx)
    # The SPHENIX self-run energy bound does not apply to MFM; it is replaced by
    # check 11 below. Everything else applies as stated.
    for c in base:
        if "max |E / E0 - 1|" in c.name or "max |P - P0|" in c.name:
            continue
        checks.append(c)

    def momentum_series(run):
        snaps = snapshots(ctx.path(run))
        p = np.array([rd.totals(s)[1] for s in snaps])
        m_w = snaps[0].read(0, "Masses", cgs=False)[-rd.N_WIND:]
        return np.linalg.norm(p - p[0], axis=1) / np.sum(m_w * rd.V_WIND)

    p_self = momentum_series("self")
    for run in ("host", "host_drift_all"):
        p_run = momentum_series(run)
        checks.append(Check(f"{run}: |P - P0| / sum m_w |v_w| at the end", float(p_run[-1]),
                            1e-5, kind="max", source="end of run, every particle "
                            "synchronised; a momentum change left in v_full would be "
                            "undone at the host's kick, an error of order 0.1"))
        checks.append(Check(f"{run}: max over snapshots of |P - P0| / sum m_w |v_w|",
                            float(p_run.max()), 2.0 * float(p_self.max()), kind="max",
                            source="twice the self run's maximum, "
                            f"{p_self.max():.3g}: snapshot velocities of particles in "
                            "mid-step are drifted by a dt/2, not synchronised"))

    def energy_series(run):
        snaps = snapshots(ctx.path(run))
        e0 = rd.totals(snaps[0])[2]
        return np.array([rd.totals(s)[2] / e0 - 1 for s in snaps])

    e_self = energy_series("self")
    checks.append(Check("self (MFM): max |E / E0 - 1| (recorded)", float(np.abs(e_self).max()),
                        1e-2, kind="max", source="MFM total-energy baseline, no merge"))
    for run in ("host", "host_drift_all"):
        e = energy_series(run)
        n = min(len(e), len(e_self))
        checks.append(Check(f"{run}: max |dE/E0 - dE_self/E0|",
                            float(np.abs(e[:n] - e_self[:n]).max()), TOL_E_MERGE,
                            kind="max", source="merge turns relative kinetic energy into "
                            "heat and conserves the total"))
        first, last = snapshots(ctx.path(run))[0], snapshots(ctx.path(run))[-1]
        m0 = first.read(0, "Masses", cgs=False)
        m_base = float(np.median(m0))
        m1 = last.read(0, "Masses", cgs=False)
        excess = float(np.sum(m1[m1 > 1.5 * m_base] - m_base))
        checks.append(Check(f"{run}: host mass excess / merged wind mass",
                            excess / (rd.N_WIND * m_base), 1.0, kind="rel", tol=1e-6,
                            source="each merged particle's mass is in its host"))
    return checks
