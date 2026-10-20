"""GIZMO-MFM: EAGLE stochastic thermal feedback fires the expected number of events.

feedback/energy_budget_thermal built with GIZMO-MFM: 128 stars of 20 gas
masses in uniform gas, Delta T = 1e7 K, 50 Myr. Checks as there: energy
gained per counted event against dE_heat (DVS12 eq. 2, 5 per cent for the
tracer's undercount), and the number of events and the energy gained against
the CCSN budget within 4 Poisson sigma of about 7300 events. With a single
star, one or two events are both within Poisson scatter of the expectation;
this test resolves the event rate to 1.2 per cent. Measured 2026-10-20
(gizmo-mfm): 7217 events against 7260 expected (0.5 sigma), energy per
event 1.017 dE_heat, energy gained 1.011 E_SN (0.9 sigma). The event rate
in MFM is therefore that of the CCSN budget; one against two events for a
single star is sampling. The run takes about 700 s on 8 threads.
"""

from ..feedback.energy_budget_thermal import DELTA_T, analyze, run  # noqa: F401

DESCRIPTION = "GIZMO-MFM: EAGLE thermal SN feedback event count and energy against the CCSN budget"
TIER = "long"
BUILDS = {"thermal": ["--with-subgrid=EAGLE", "--with-ext-potential=constant",
                      "--with-number-of-SNII-rays=8",
                      "--with-hydro=gizmo-mfm", "--with-riemann-solver=hllc"]}
