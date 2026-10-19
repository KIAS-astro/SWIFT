"""GIZMO: gas next to an enriching star must cool like gas far from it.

A uniform periodic box of hot gas (n_H = 0.1 cm^-3, T = 1e6 K, solar
metallicity) cools radiatively (EAGLE tables, z = 0). Star particles born
at t = 0 sit in it; while younger than EAGLEFeedback:stellar_evolution_age_cut
their enrichment calls the EAGLE feedback loop for every gas neighbour at
every star step, and that loop rewrites the
neighbour's thermal energy from hydro_get_physical_internal_energy()
(feedback/EAGLE_thermal/feedback_iact.h, the u_new_enrich update). SNII
feedback is off, so no neighbour should be heated measurably: the enrichment
energy (kinetic energy of the ejecta in the gas frame, here zero, plus the
enrichment terms) changes the gas temperature by far less than cooling does.

dt_max = 1e-5 internal units (9.8 kyr) puts every particle on the same time
bin, so each gas step coincides with a step of its neighbouring stars.

Before the fix, GIZMO's getter returned P/rho, which after kick2 still holds
the start-of-step energy, so each enrichment call wrote back the energy
before the step's cooling. Star neighbours then never cooled. Measured 2026-10-20
(gizmo-mfm, HLLC, 8 threads): before the fix <u>_near / <u>_far = 164 at
5 Myr, and the run aborted at 9.7 Myr (smoothing lengths outgrew the box);
after it 0.943, against 0.941 for SPHENIX (SWIFT_REG_HYDRO=sphenix).

Checks: the mean internal energy of gas within 1.5 h of a star over that of
gas farther than 4 h from every star, at the end (t = 20 Myr, about 10
cooling times at 1e6 K): equal within 30 per cent. The scatter of the far
gas between snapshots is under 5 per cent; 30 per cent allows for the
genuine enrichment and the slightly higher metallicity next to stars, and
lies far below the factor ~100 of the failure.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "GIZMO: star neighbours cool like other gas (feedback must not undo cooling)"
TIER = "quick"
HYDRO = os.environ.get("SWIFT_REG_HYDRO", "gizmo-mfm")
BUILDS = {"eagle": ["--with-subgrid=EAGLE", "--with-ext-potential=constant"]
          + ([] if HYDRO == "sphenix" else
             [f"--with-hydro={HYDRO}", "--with-riemann-solver=hllc"])}

N = 16
BOX = 1.0  # kpc
NH = 0.1
T0 = 1e6
N_STARS = 16
SEED = 7
T_END = 0.02045  # internal units, 20 Myr
DT_MAX = 1e-5


def run(ctx):
    gas, _ = ics.uniform_gas(N, BOX, NH, T0)
    rng = np.random.default_rng(SEED)
    stars = {
        "Coordinates": rng.uniform(0, BOX, size=(N_STARS, 3)),
        "Velocities": np.zeros((N_STARS, 3)),
        "Masses": np.full(N_STARS, gas["Masses"][0]),
        "SmoothingLength": np.full(N_STARS, 1.2348 * BOX / N),
        "ParticleIDs": np.arange(N ** 3 + 1, N ** 3 + 1 + N_STARS, dtype=np.uint64),
    }
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas, stars=stars)
    for t in ("photometry", "coolingtables", "yieldtables"):
        ctx.table(t)
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END / 4, dt_max=DT_MAX)
    p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
    # stars born at t = 0: younger than stellar_evolution_age_cut, so they
    # enrich at every one of their steps rather than every sampling_rate-th
    p["Stars"].update({"overwrite_birth_time": 1, "birth_time": 0.0})
    p["EAGLEFeedback"].update({"use_SNII_feedback": 0})
    p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                   "Cool_temperature_norm_K": 10.0})
    ctx.swift(p, ["--hydro", "--stars", "--feedback", "--cooling",
                  "--external-gravity", "--limiter", "--sync"], timeout=1800)


def analyze(ctx):
    last = snapshots(ctx.run_dir)[-1]
    xg = last.read(0, "Coordinates")
    hg = last.read(0, "SmoothingLengths")
    u = last.read(0, "InternalEnergies")
    xs = last.read(4, "Coordinates")
    box = BOX * units.UNIT_LENGTH
    d = np.full(len(xg), np.inf)
    for s in xs:
        dx = np.abs(xg - s)
        dx = np.minimum(dx, box - dx)
        d = np.minimum(d, np.linalg.norm(dx, axis=1))
    near = d < 1.5 * hg
    far = d > 4.0 * hg
    ratio = u[near].mean() / u[far].mean()
    ctx.info.update(n_near=int(near.sum()), n_far=int(far.sum()),
                    u_near=float(u[near].mean()), u_far=float(u[far].mean()),
                    u0=units.u_from_T(T0, units.MU_IONISED))
    return [
        Check("<u> near stars / <u> far from stars", ratio, 1.0, kind="rel", tol=0.3,
              source="same gas, same cooling; enrichment energy < 1e-6 per step"),
        Check("far gas cooled: <u>_far / u(1e6 K)", u[far].mean() / ctx.info["u0"], 0.1,
              kind="max", source="cooling time at 1e6 K, n_H = 0.1, Z_sun is ~2 Myr"),
    ]
