"""GIZMO-MFM: EAGLE kinetic SN feedback must reach the gas momentum.

Same set-up and expected value as feedback/energy_budget_kinetic (uniform
gas, 128 stars, f_kin = 1, Delta v = 600 km/s, 50 Myr; energy gained by the
gas = E_SN within 4 Poisson sigma of the number of kicked pairs), built with
GIZMO-MFM.

Kinetic feedback kicks a gas particle by adding to xp->v_full. GIZMO's
conserved momentum is what the particle velocity is reset from at every
kick (hydro_velocities_set), so before the fix every feedback kick was
erased at the next kick and only the heat of unkicked pairs reached the
gas. Measured 2026-10-20 (gizmo-mfm, HLLC, 8 threads): energy gained / E_SN
= 0.043 before the fix, 1.013 after it (0.8 sigma); the net
momentum fraction is 6.9e-5 before and 9.6e-8 after. The run takes
about 1000 s on 8 threads, hence the long tier.

A second check: MFM fluxes are pairwise antisymmetric and the kicks come in
equal and opposite pairs, so the net gas momentum stays small against
sum m|v|; tolerance 2 per cent (SPHENIX reaches 2.9 per cent in the thermal
run, see feedback/energy_budget_kinetic).
"""

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..feedback import _common as c

DESCRIPTION = "GIZMO-MFM: EAGLE kinetic feedback kicks reach the gas (energy = CCSN budget)"
TIER = "long"
BUILDS = {"kinetic": [
    "--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
    "--with-entropy-floor=EAGLE", "--with-stars=EAGLE", "--with-star-formation=EAGLE",
    "--with-feedback=EAGLE-kinetic", "--with-black-holes=EAGLE",
    "--with-ext-potential=constant", "--with-number-of-SNII-rays=8",
    "--with-hydro=gizmo-mfm", "--with-riemann-solver=hllc",
]}

DELTA_V = 600.0  # km/s


def run(ctx):
    p, mg, ms = c.setup(ctx, {"SNII_delta_v_km_p_s": DELTA_V})
    ctx.info["m_gas"], ctx.info["m_star"] = mg, ms
    ctx.swift(p, c.FLAGS, timeout=3600)


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    first, last = snaps[0], snaps[-1]
    fb = ctx.info["feedback"]
    mg = ctx.info["m_gas"]

    gained = c.total_energy(last) - c.total_energy(first)
    budget = c.expected_sn_energy(fb, ctx.info["m_star"])
    n_pairs = budget / (mg * (DELTA_V * 1e5) ** 2)
    m = last.read(0, "Masses")
    v = last.read(0, "Velocities")
    p_net = np.linalg.norm(c.gas_momentum(last))
    p_abs = np.sum(m * np.linalg.norm(v, axis=1))
    ctx.info.update(gained_erg=gained, budget_erg=budget, expected_pairs=n_pairs,
                    net_momentum_fraction=p_net / p_abs)
    return [
        Check("energy gained / CCSN budget", gained / budget, 1.0, kind="sigma", tol=4.0,
              sigma=1.0 / np.sqrt(n_pairs),
              source="f_E = 1, f_kin = 1; Poisson scatter of the number of pairs"),
        Check("net gas momentum / sum m|v|", p_net / p_abs, 0.02, kind="max",
              source="antisymmetric MFM fluxes, pairwise opposite kicks"),
    ]
