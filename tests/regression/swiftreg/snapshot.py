"""Read SWIFT snapshots and convert fields to cgs.

Every SWIFT output dataset carries the attribute
"Conversion factor to physical CGS (including cosmological corrections)",
so values read here do not depend on the internal units of the run.
"""

import glob
import os

import h5py
import numpy as np

CGS_ATTR = "Conversion factor to physical CGS (including cosmological corrections)"


class Snapshot:
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.time = float(f["Header"].attrs["Time"][0]
                              if np.ndim(f["Header"].attrs["Time"])
                              else f["Header"].attrs["Time"])
            self.time_cgs = self.time * float(
                f["Units"].attrs["Unit time in cgs (U_t)"][0]
            )
            self.npart = np.array(f["Header"].attrs["NumPart_ThisFile"])

    def has(self, ptype, name):
        with h5py.File(self.path, "r") as f:
            return f"PartType{ptype}/{name}" in f

    def read(self, ptype, name, cgs=True):
        """Return a field of PartType<ptype>, in cgs unless cgs=False."""
        with h5py.File(self.path, "r") as f:
            group = f"PartType{ptype}"
            if group not in f:
                return np.zeros(0)
            dset = f[f"{group}/{name}"]
            data = dset[...]
            if cgs and CGS_ATTR in dset.attrs:
                data = data * float(dset.attrs[CGS_ATTR][0])
            return data


def snapshots(run_dir, basename="snap"):
    """All snapshots of a run in output order."""
    paths = sorted(glob.glob(os.path.join(run_dir, f"{basename}_[0-9]*.hdf5")))
    return [Snapshot(p) for p in paths]
