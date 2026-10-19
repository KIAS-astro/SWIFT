"""Shared set-up for the mixed-mass hydro tests (L-T1, L-T2 of
light-particles-design.md Sec. 6): plain hydro builds, cgs internal units
(every unit 1), so code values are the dimensionless values of the problem.
"""

import h5py
import numpy as np

GAMMA = 5.0 / 3.0
RATIOS = [1, 3, 10, 30]
SCHEMES = {"sph": "sphenix", "mfm": "gizmo-mfm"}


def build(scheme, dim=3):
    flags = [f"--with-hydro={scheme}"]
    if dim != 3:
        flags.append(f"--with-hydro-dimension={dim}")
    if scheme.startswith("gizmo"):
        flags.append("--with-riemann-solver=hllc")
    return flags


def flags(scheme):
    """Run flags: GIZMO needs the time-step limiter (Hopkins 2015, app. G)."""
    f = ["--hydro", "--limiter"]
    if scheme.startswith("gizmo"):
        f.append("--sync")
    return f


def write_ic(path, x, v, m, u, h, box, dim):
    n = len(m)
    with h5py.File(path, "w") as f:
        g = f.create_group("Header")
        g.attrs["BoxSize"] = box
        g.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        g.attrs["NumPart_Total_HighWord"] = [0] * 6
        g.attrs["NumPart_ThisFile"] = [n, 0, 0, 0, 0, 0]
        g.attrs["Time"] = 0.0
        g.attrs["NumFilesPerSnapshot"] = 1
        g.attrs["MassTable"] = [0.0] * 6
        g.attrs["Flag_Entropy_ICs"] = 0
        g.attrs["Dimension"] = dim
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=x)
        g.create_dataset("Velocities", data=v, dtype="f")
        g.create_dataset("Masses", data=m, dtype="f")
        g.create_dataset("SmoothingLength", data=h, dtype="f")
        g.create_dataset("InternalEnergy", data=u, dtype="f")
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1), dtype="L")


def params(t_end, n_snap=1, cfl=0.1, dt_min=1e-10, dt_max=1e-2, box=None,
           top_cells=None):
    unit = {k: 1.0 for k in ["UnitMass_in_cgs", "UnitLength_in_cgs",
                             "UnitVelocity_in_cgs", "UnitCurrent_in_cgs",
                             "UnitTemp_in_cgs"]}
    p = {
        "InternalUnitSystem": unit,
        "TimeIntegration": {"time_begin": 0.0, "time_end": float(t_end),
                            "dt_min": dt_min, "dt_max": dt_max},
        "Snapshots": {"basename": "snap", "time_first": 0.0,
                      "delta_time": float(t_end) / n_snap},
        "Statistics": {"delta_time": float(t_end), "time_first": 0.0},
        "SPH": {"resolution_eta": 1.2348, "CFL_condition": cfl},
        "InitialConditions": {"file_name": "ics.hdf5", "periodic": 1},
    }
    if top_cells is not None:
        p["Scheduler"] = {"max_top_level_cells": top_cells}
    return p
