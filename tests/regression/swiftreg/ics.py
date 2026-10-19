"""Write SWIFT initial-condition files in the test internal units.

The files carry no /Units group, so SWIFT reads them in the internal unit
system of the parameter file, which every test takes from units.py.
"""

import h5py
import numpy as np

from . import units


def lattice(n, box_size):
    """Cell-centred cubic lattice of n^3 positions filling a periodic box."""
    dx = box_size / n
    x = (np.arange(n) + 0.5) * dx
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def uniform_gas(n, box_size_kpc, nH_cgs, T, mu=units.MU_NEUTRAL):
    """Gas fields for a uniform periodic box on an n^3 lattice.

    Returns a dictionary of arrays in internal units and the gas particle
    mass in solar masses.
    """
    rho = units.rho_from_nH(nH_cgs) / units.UNIT_DENSITY
    npart = n ** 3
    mass = rho * box_size_kpc ** 3 / npart
    u = units.u_from_T(T, mu) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": lattice(n, box_size_kpc),
        "Velocities": np.zeros((npart, 3)),
        "Masses": np.full(npart, mass),
        "SmoothingLength": np.full(npart, 1.2348 * box_size_kpc / n),
        "InternalEnergy": np.full(npart, u),
        "ParticleIDs": np.arange(1, npart + 1, dtype=np.uint64),
    }
    return gas, mass * units.UNIT_MASS / units.SOLAR_MASS


def write(path, box_size, gas=None, stars=None):
    """Write gas (PartType0) and star (PartType4) dictionaries to path."""
    parts = {0: gas, 4: stars}
    counts = [0] * 7
    for ptype, fields in parts.items():
        if fields is not None:
            counts[ptype] = len(fields["ParticleIDs"])

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = np.full(3, box_size, dtype=np.float64)
        h.attrs["NumPart_Total"] = np.array(counts, dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(7, dtype=np.uint32)
        h.attrs["NumPart_ThisFile"] = np.array(counts, dtype=np.uint32)
        h.attrs["Time"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["MassTable"] = np.zeros(7)
        h.attrs["Flag_Entropy_ICs"] = 0
        h.attrs["Dimension"] = 3

        for ptype, fields in parts.items():
            if fields is None:
                continue
            g = f.create_group(f"PartType{ptype}")
            for name, data in fields.items():
                g.create_dataset(name, data=data)
