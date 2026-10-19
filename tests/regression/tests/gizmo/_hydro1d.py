"""Shared set-up for the 1D GIZMO hydro tests: ICs, parameters, exact Riemann solution.

The runs use a 1D build (--with-hydro-dimension=1) and cgs internal units
(every unit 1), so code values are the dimensionless values of the problem.
"""

import h5py
import numpy as np
from scipy.optimize import brentq

GAMMA = 5.0 / 3.0


def build(scheme="gizmo-mfm", riemann="hllc"):
    flags = [f"--with-hydro={scheme}", "--with-hydro-dimension=1"]
    if scheme.startswith("gizmo"):
        flags.append(f"--with-riemann-solver={riemann}")
    return flags


def write_ic(path, x, v, m, u, h, box):
    n = len(x)
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
        g.attrs["Dimension"] = 1
        g = f.create_group("PartType0")
        c = np.zeros((n, 3))
        c[:, 0] = x
        vv = np.zeros((n, 3))
        vv[:, 0] = v
        g.create_dataset("Coordinates", data=c)
        g.create_dataset("Velocities", data=vv, dtype="f")
        g.create_dataset("Masses", data=m, dtype="f")
        g.create_dataset("SmoothingLength", data=h, dtype="f")
        g.create_dataset("InternalEnergy", data=u, dtype="f")
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1), dtype="L")


def params(t_end, cfl=0.1, dt_min=1e-8, dt_max=1e-2):
    unit = {k: 1.0 for k in ["UnitMass_in_cgs", "UnitLength_in_cgs",
                             "UnitVelocity_in_cgs", "UnitCurrent_in_cgs",
                             "UnitTemp_in_cgs"]}
    return {
        "InternalUnitSystem": unit,
        "TimeIntegration": {"time_begin": 0.0, "time_end": float(t_end),
                            "dt_min": dt_min, "dt_max": dt_max},
        "Snapshots": {"basename": "snap", "time_first": 0.0,
                      "delta_time": float(t_end)},
        "Statistics": {"delta_time": float(t_end), "time_first": 0.0},
        "SPH": {"resolution_eta": 1.2348, "CFL_condition": cfl},
        "InitialConditions": {"file_name": "ics.hdf5", "periodic": 1},
    }


def read(snap):
    """Arrays x, v, rho, P of the last snapshot (code units = cgs here)."""
    return {"x": snap.read(0, "Coordinates")[:, 0], "v": snap.read(0, "Velocities")[:, 0],
            "rho": snap.read(0, "Densities"), "P": snap.read(0, "Pressures")}


def _f(p, r, P):
    if p > P:
        A = 2 / ((GAMMA + 1) * r)
        B = (GAMMA - 1) / (GAMMA + 1) * P
        return (p - P) * np.sqrt(A / (p + B))
    a = np.sqrt(GAMMA * P / r)
    return 2 * a / (GAMMA - 1) * ((p / P) ** ((GAMMA - 1) / (2 * GAMMA)) - 1)


def riemann_exact(WL, WR, xi):
    """Exact solution (rho, v, P) of the Riemann problem at x/t = xi (Toro 2009, ch. 4)."""
    G = GAMMA
    rL, uL, PL = WL
    rR, uR, PR = WR
    aL, aR = np.sqrt(G * PL / rL), np.sqrt(G * PR / rR)
    ps = brentq(lambda p: _f(p, rL, PL) + _f(p, rR, PR) + uR - uL, 1e-14, 1e14)
    us = 0.5 * (uL + uR) + 0.5 * (_f(ps, rR, PR) - _f(ps, rL, PL))
    gm = (G - 1) / (G + 1)
    out = np.zeros((len(xi), 3))
    for i, s in enumerate(xi):
        left = s <= us
        r0, u0, P0, a0, sg = (rL, uL, PL, aL, -1) if left else (rR, uR, PR, aR, 1)
        if ps > P0:
            S = u0 + sg * a0 * np.sqrt((G + 1) / (2 * G) * ps / P0 + (G - 1) / (2 * G))
            outside = s <= S if left else s >= S
            out[i] = (r0, u0, P0) if outside else (
                r0 * (ps / P0 + gm) / (gm * ps / P0 + 1), us, ps)
        else:
            a_s = a0 * (ps / P0) ** ((G - 1) / (2 * G))
            head, tail = u0 + sg * a0, us + sg * a_s
            if (left and s <= head) or (not left and s >= head):
                out[i] = (r0, u0, P0)
            elif (left and s >= tail) or (not left and s <= tail):
                out[i] = (r0 * (ps / P0) ** (1 / G), us, ps)
            else:
                u = 2 / (G + 1) * (-sg * a0 + (G - 1) / 2 * u0 + s)
                a = 2 / (G + 1) * (a0 - sg * (G - 1) / 2 * (u0 - s))
                out[i] = (r0 * (a / a0) ** (2 / (G - 1)), u,
                          P0 * (a / a0) ** (2 * G / (G - 1)))
    return out


def shocktube_ics(path, n_left, WL, WR):
    """Two mirrored tubes in a periodic box [0, 2): left state in [0.5, 1.5),
    equal-mass particles; the tube analysed has its interface at x = 1.5."""
    rL, rR = WL[0], WR[0]
    nR = max(int(round(n_left * rR / rL)), 4)
    x = np.concatenate([0.5 + (np.arange(n_left) + 0.5) / n_left,
                        np.mod(1.5 + (np.arange(nR) + 0.5) / nR, 2.0)])
    m = np.full(len(x), rL / n_left)
    rho = np.concatenate([np.full(n_left, rL), np.full(nR, rR)])
    P = np.concatenate([np.full(n_left, WL[2]), np.full(nR, WR[2])])
    v = np.concatenate([np.full(n_left, WL[1]), np.full(nR, WR[1])])
    h = 1.2348 * m / rho
    write_ic(path, x, v, m, P / ((GAMMA - 1) * rho), h, 2.0)
