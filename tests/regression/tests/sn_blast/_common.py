"""Shared set-up for the isolated supernova tests.

A uniform periodic box of gas at rest; the SN energy is put into the initial
conditions as internal energy of the N_INJECT particles nearest the box centre,
as in examples/HydroTests/SedovBlast_3D. Measurements are made about the centre.
"""

import numpy as np

from swiftreg import ics, units

E_SN = 1e51  # erg
N_INJECT = 32  # the 8 + 24 lattice sites nearest the centre of an even lattice


def blast_ics(path, n, box_kpc, nH, T, glass_file=None):
    """Write the ICs; return (gas mass in g, background density in g/cm^3).
    With glass_file the particles sit on that glass instead of a lattice."""
    gas, _ = ics.uniform_gas(n, box_kpc, nH, T, glass_file=glass_file)
    r = np.linalg.norm(gas["Coordinates"] - 0.5 * box_kpc, axis=1)
    inject = np.argsort(r, kind="stable")[:N_INJECT]
    m = gas["Masses"][0]
    gas["InternalEnergy"][inject] += (E_SN / units.UNIT_ENERGY_PER_MASS / units.UNIT_MASS
                                      / (N_INJECT * m))
    ics.write(path, box_kpc, gas=gas)
    return m * units.UNIT_MASS, units.rho_from_nH(nH)


def radial(snap, box_cm):
    """Radius from the centre (cm), mass, radial velocity, density, u, T if present."""
    x = snap.read(0, "Coordinates") - 0.5 * box_cm
    r = np.linalg.norm(x, axis=1)
    v = snap.read(0, "Velocities")
    vr = np.sum(v * x, axis=1) / np.maximum(r, 1e-30)
    out = {
        "r": r,
        "m": snap.read(0, "Masses"),
        "vr": vr,
        "v": v,
        "rho": snap.read(0, "Densities"),
        "u": snap.read(0, "InternalEnergies"),
        "P": snap.read(0, "Pressures"),
    }
    if snap.has(0, "Temperatures"):
        out["T"] = snap.read(0, "Temperatures")
    return out


def binned(r, q, edges, weights=None):
    """Mean of q in radial bins (weighted by weights if given)."""
    w = np.ones_like(r) if weights is None else weights
    num, _ = np.histogram(r, edges, weights=w * q)
    den, _ = np.histogram(r, edges, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den
