"""A uniform periodic box at rest must stay uniform and at rest.

This checks the harness end to end (build, ICs, parameters, run, snapshot
reading, units) on the simplest problem with an exact answer: every
particle of a cubic lattice has the same SPH density, so pressure forces
cancel and nothing moves.
"""

import numpy as np

from swiftreg import ics, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "uniform box at rest stays uniform and at rest (harness check)"
TIER = "quick"
BUILDS = {"hydro": []}

N = 16
BOX = 1.0  # kpc
NH = 0.1  # cm^-3
T = 1e4  # K
T_END = 1e-2  # internal time units, about 9.8 Myr


def run(ctx):
    gas, _ = ics.uniform_gas(N, BOX, NH, T)
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas)
    p = {
        "InternalUnitSystem": units.internal_unit_params(),
        "TimeIntegration": {"time_begin": 0.0, "time_end": T_END,
                            "dt_min": 1e-12, "dt_max": 1e-3},
        "Snapshots": {"basename": "snap", "time_first": 0.0,
                      "delta_time": T_END},
        "Statistics": {"delta_time": T_END, "time_first": 0.0},
        "InitialConditions": {"file_name": "ics.hdf5", "periodic": 1},
        "SPH": {"resolution_eta": 1.2348, "CFL_condition": 0.1,
                "minimal_temperature": 10.0},
    }
    ctx.swift(p, ["--hydro"])


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    last = snaps[-1]
    rho = last.read(0, "Densities")
    v = last.read(0, "Velocities")
    u = last.read(0, "InternalEnergies")

    rho_expected = units.rho_from_nH(NH)
    cs = units.sound_speed(units.u_from_T(T, units.MU_NEUTRAL))
    vmax = np.max(np.linalg.norm(v, axis=1))
    ctx.info["final_time_Myr"] = last.time_cgs / units.MYR

    return [
        Check("snapshots written", len(snaps), 2, kind="min",
              source="Snapshots:delta_time = time_end"),
        Check("final time [Myr]", last.time_cgs / units.MYR,
              T_END * units.UNIT_TIME / units.MYR, kind="rel", tol=1e-6,
              source="TimeIntegration:time_end"),
        Check("mean density [g cm^-3]", rho.mean(), rho_expected, kind="rel",
              tol=1e-2,
              source="lattice mass / volume; SPH estimate on a lattice is "
                     "biased by under 1 per cent for eta = 1.2348"),
        Check("density spread (max-min)/mean", np.ptp(rho) / rho.mean(), 1e-5,
              kind="max", source="identical neighbourhoods on a lattice"),
        Check("max speed / sound speed", vmax / cs, 1e-6, kind="max",
              source="pressure forces cancel by symmetry"),
        Check("internal energy unchanged", u.mean(),
              units.u_from_T(T, units.MU_NEUTRAL), kind="rel", tol=1e-5,
              source="adiabatic, no forces"),
    ]
