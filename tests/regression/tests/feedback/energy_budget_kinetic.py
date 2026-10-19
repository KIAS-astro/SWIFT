"""EAGLE kinetic feedback injects the CCSN energy budget.

The EAGLE-kinetic module (Chaikin et al. 2023 with f_kin = 1) kicks gas
particles in pairs, in opposite directions, with a velocity correction that
conserves energy and linear momentum exactly. With equal gas masses the
energy per pair is dE_pair = m_g Delta v^2, so the number of pairs over the
run is E_SN / dE_pair = 4200 for Delta v = 600 km/s, a 1.5 per cent Poisson
scatter. Pairs that cannot be kicked (both members not granted to the star)
put their energy into the gas as heat, so the total energy gained still
counts it.

Check: energy gained by the gas (thermal + kinetic) = E_SN within 4 sigma.

Momentum is not checked here. Each pair is kicked with equal and opposite
momenta, but SPH with individual time-steps and the time-step limiter does
not conserve momentum exactly: in the thermal-feedback run, which injects no
momentum at all, the net gas momentum reached 2.9 per cent of sum m|v| after
50 Myr (2026-10-20). The net momentum is recorded in the results for
information; pair conservation is tested directly on the kick function.
"""

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "EAGLE kinetic SN feedback: injected energy equals the CCSN budget"
TIER = "quick"
BUILDS = {"kinetic": [
    "--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
    "--with-entropy-floor=EAGLE", "--with-stars=EAGLE", "--with-star-formation=EAGLE",
    "--with-feedback=EAGLE-kinetic", "--with-black-holes=EAGLE",
    "--with-ext-potential=constant", "--with-number-of-SNII-rays=8",
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
    dE_pair = mg * (DELTA_V * 1e5) ** 2
    n_pairs = budget / dE_pair

    m = last.read(0, "Masses")
    v = last.read(0, "Velocities")
    p_net = np.linalg.norm(c.gas_momentum(last))
    p_abs = np.sum(m * np.linalg.norm(v, axis=1))
    ctx.info.update(gained_erg=gained, budget_erg=budget, expected_pairs=n_pairs,
                    net_momentum_fraction=p_net / p_abs)

    return [
        Check("energy gained / CCSN budget", gained / budget, 1.0, kind="sigma", tol=4.0,
              sigma=1.0 / np.sqrt(n_pairs),
              source="f_E = 1, f_kin = 1; Poisson scatter of the number of pairs "
                     "E_SN / (m_g dv^2)"),
    ]
