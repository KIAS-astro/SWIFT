"""GIZMO: the entropy getter agrees with the internal-energy getter.

A uniform periodic box of hot gas (n_H = 0.1 cm^-3, T = 1e6 K, solar
metallicity) cools radiatively (EAGLE tables, z = 0) for 0.5 Myr with
dt_max = 1e-5 internal units (9.8 kyr). The mean u falls by a factor of
about 100 over the run, so each step removes several per cent of the
thermal energy; by 1 Myr the gas would sit on the 8000 K floor and the
getters would agree again whatever their bug.

The non-drifted getters must describe the same thermodynamic state:
A = (gamma - 1) u / rho^(gamma - 1). hydro_get_comoving_internal_energy
returns the conserved E/m; hydro_get_comoving_entropy returned P/rho^gamma,
and P is recomputed from E only in the next density loop, so after kick2
it still holds the energy of the start of the step (plus its drift
prediction). EAGLE star formation reads this entropy after kick2
(star_formation/EAGLE/star_formation.h), and the snapshot field Entropies
is written by the same getter (hydro_io.h, convert_A), as InternalEnergies
is by the internal-energy getter (convert_u). The check therefore compares
the two snapshot fields.

Measured 2026-10-20 (gizmo-mfm, HLLC, 8 threads): max |A / A(u) - 1| =
0.016-0.157 at the five snapshots before the fix, at most 2.5e-7 after it.

Check: at every snapshot after t = 0, max over particles of
|A / A(u) - 1| <= 1e-4. Both fields are single precision and A(u) involves
one pow(), so agreement is to a few times 1e-7; one step of cooling gives
a difference of order 1e-2.
"""

import numpy as np

from swiftreg import ics, params
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "GIZMO: snapshot Entropies agree with InternalEnergies in cooling gas"
TIER = "quick"
BUILDS = {"eagle": ["--with-subgrid=EAGLE",
                    "--with-hydro=gizmo-mfm", "--with-riemann-solver=hllc"]}

N = 16
BOX = 1.0  # kpc
NH = 0.1
T0 = 1e6
T_END = 5.11e-4  # internal units, 0.5 Myr
DT_MAX = 1e-5
GAMMA = 5.0 / 3.0


def run(ctx):
    gas, _ = ics.uniform_gas(N, BOX, NH, T0)
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas)
    for t in ("photometry", "coolingtables", "yieldtables"):
        ctx.table(t)
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END / 5, dt_max=DT_MAX)
    ctx.swift(p, ["--hydro", "--cooling", "--temperature"], timeout=600)


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    out = []
    u0 = float(np.mean(snaps[0].read(0, "InternalEnergies", cgs=False)))
    for s in snaps[1:]:
        rho = s.read(0, "Densities", cgs=False)
        u = s.read(0, "InternalEnergies", cgs=False)
        A = s.read(0, "Entropies", cgs=False)
        A_u = (GAMMA - 1.0) * u / rho ** (GAMMA - 1.0)
        err = float(np.max(np.abs(A / A_u - 1.0)))
        t_myr = s.time_cgs / 3.15576e13
        ctx.info[f"max_rel_diff t = {t_myr:.2f} Myr"] = err
        ctx.info[f"mean_u_over_initial t = {t_myr:.2f} Myr"] = float(np.mean(u)) / u0
        out.append(Check(f"max |A / A(u) - 1| at t = {t_myr:.2f} Myr", err, 1e-4, kind="max",
                         source="A = (gamma - 1) u / rho^(gamma - 1), same particle state"))
    return out
