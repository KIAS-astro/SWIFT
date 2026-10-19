"""GIZMO: particle splitting conserves momentum and energy (hydro_set_mass).

A uniform periodic box (16^3 lattice, n_H = 0.1 cm^-3, T = 1e4 K) moving
with a bulk velocity of 100 km/s along x. SPH:particle_splitting is on with
a threshold below the gas mass, so every particle splits into two halves at
the first rebuild (engine_split_particles.c, which calls hydro_set_mass on
both halves). A split must keep each particle's velocity and specific
internal energy, so total momentum and energy are unchanged.

GIZMO's hydro_set_mass changed only conserved.mass, while velocity and
specific energy are conserved momentum and energy divided by the mass:
each half kept the parent's full momentum and energy. Measured 2026-10-20
(gizmo-mfm): total energy / initial = 1.0198 before the fix (the thermal
energy, 2.4 per cent of the total here, doubled), momentum exact only
because the previous commit moves the v_full mismatch into the momentum.
After: 1 within 2e-7; MFV and SPHENIX 1 within 4e-7.

Checks after 1 Myr: particle number doubled; total momentum and total
(kinetic + thermal) energy equal to the initial values within 1e-4 (float
round-off over 8192 particles; a uniform box has no other source of change).
"""

import os

import numpy as np

from swiftreg import ics, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "GIZMO: particle splitting keeps v and u (momentum and energy conserved)"
TIER = "quick"
HYDRO = os.environ.get("SWIFT_REG_HYDRO", "gizmo-mfm")
BUILDS = {"hydro": [] if HYDRO == "sphenix" else
          [f"--with-hydro={HYDRO}", "--with-riemann-solver=hllc"]}

N = 16
BOX = 1.0
NH = 0.1
T0 = 1e4
V_BULK = 100.0  # km/s
T_END = 1.0227e-3  # 1 Myr


def run(ctx):
    gas, _ = ics.uniform_gas(N, BOX, NH, T0)
    gas["Velocities"][:, 0] = V_BULK
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas)
    m = gas["Masses"][0]
    p = {
        "InternalUnitSystem": units.internal_unit_params(),
        "TimeIntegration": {"time_begin": 0.0, "time_end": T_END,
                            "dt_min": 1e-12, "dt_max": 1e-4},
        "Snapshots": {"basename": "snap", "time_first": 0.0, "delta_time": T_END},
        "Statistics": {"delta_time": T_END, "time_first": 0.0},
        "Scheduler": {"max_top_level_cells": 8},
        "InitialConditions": {"file_name": "ics.hdf5", "periodic": 1},
        "SPH": {"resolution_eta": 1.2348, "CFL_condition": 0.1,
                "minimal_temperature": 10.0, "particle_splitting": 1,
                "particle_splitting_mass_threshold": float(0.75 * m)},
    }
    ctx.swift(p, ["--hydro", "--limiter"], timeout=600)


def totals(s):
    m = s.read(0, "Masses")
    v = s.read(0, "Velocities")
    u = s.read(0, "InternalEnergies")
    return (len(m), np.sum(m * v[:, 0]),
            np.sum(m * u) + 0.5 * np.sum(m * np.sum(v * v, axis=1)))


def analyze(ctx):
    snaps = snapshots(ctx.run_dir)
    n0, p0, e0 = totals(snaps[0])
    n1, p1, e1 = totals(snaps[-1])
    ctx.info.update(n0=n0, n1=n1, p_ratio=p1 / p0, e_ratio=e1 / e0)
    return [
        Check("particle number after splitting", n1, 2 * N ** 3, kind="equal",
              source="threshold 0.75 m_gas: every particle splits once"),
        Check("total x-momentum / initial", p1 / p0, 1.0, kind="rel", tol=1e-4,
              source="splitting keeps velocities"),
        Check("total energy / initial", e1 / e0, 1.0, kind="rel", tol=1e-4,
              source="splitting keeps v and u; uniform box at rest in its frame"),
    ]
