"""Equilibrium of the H12 isolated disc ICs without cooling, star formation or feedback.

The ICs (M6_disk of Nobels et al. 2024, and copies subsampled by 4 and 32 to particle
masses of 3.2e6 and 2.6e7 Msun by swift-ism-model tests/isolated_galaxy/
make_ics.py) are evolved for 0.5 Gyr with hydrodynamics, self-gravity and the analytic
Hernquist halo only. A disc that starts in equilibrium keeps its profiles. Softening
0.4, 0.8 and 1.7 kpc (ai-notes isolated-galaxy-ics.md Sec. on softening).

Checks (isolated-galaxy-ics.md, equilibrium test), in annuli 2-4, 4-6 and 6-10 kpc,
between t = 0 and 0.5 Gyr: stellar surface density within 10 per cent, stellar
vertical velocity dispersion within 20 per cent, stellar rotation velocity within
5 per cent.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from .fkin_series import BUILDS, ic_file  # noqa: F401

DESCRIPTION = "H12 disc ICs (8e5, 3.2e6, 2.6e7 Msun) stay in equilibrium for 0.5 Gyr"
TIER = "long"
RUNS = {"m8e5": ("M6_disk.hdf5", 0.4), "m3e6": ("H12_m3e6_disk.hdf5", 0.8),
        "m3e7": ("H12_m3e7_disk.hdf5", 1.7),
        # the two reference resolutions: HR5/8 and HR5 (Lee et al. 2021)
        "m1.6e6": ("H12_m1.6e6_disk.hdf5", 0.5), "m1.3e7": ("H12_m1.3e7_disk.hdf5", 1.0)}
T_END = 0.5113  # 500 Myr in internal units (1 unit = 977.8 Myr)
DT_SNAP = 0.05113
ANNULI = ((2.0, 4.0), (4.0, 6.0), (6.0, 10.0))


def run(ctx):
    for t in ("coolingtables", "photometry", "yieldtables"):
        ctx.table(t)
    ic_file(ctx.data_dir)
    for name, (ic, soft) in RUNS.items():
        src = os.path.join(ctx.data_dir, "ics", ic)
        os.makedirs(ctx.path(name))
        shutil.copy(src, ctx.path(name, ic))
        p = params.eagle_base(ic, T_END, ctx.data_dir, DT_SNAP, dt_max=1e-2)
        p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
        p["Gravity"].update({"max_physical_DM_softening": soft,
                             "max_physical_baryon_softening": soft})
        p["SPH"].update({"h_min_ratio": 0.1, "h_max": 10.0, "minimal_temperature": 100.0})
        p["Scheduler"] = {"max_top_level_cells": 16, "tasks_per_cell": 200}
        p["HernquistPotential"] = {
            "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
            "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
            "bulgefraction": 0.014, "timestep_mult": 0.01, "epsilon": soft}
        ctx.swift(p, ["--hydro", "--self-gravity", "--external-gravity", "--stars",
                      "--temperature", "--limiter", "--sync"], subdir=name, name=name,
                  timeout=4 * 3600)


def profiles(snap):
    x = snap.read(4, "Coordinates") / units.KPC
    v = snap.read(4, "Velocities") / 1e5
    m = snap.read(4, "Masses") / units.SOLAR_MASS
    c = np.average(x, axis=0, weights=m)
    vc = np.average(v, axis=0, weights=m)
    x, v = x - c, v - vc
    R = np.hypot(x[:, 0], x[:, 1])
    vphi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / np.maximum(R, 1e-6)
    out = []
    for lo, hi in ANNULI:
        s = (R >= lo) & (R < hi) & (np.abs(x[:, 2]) < 2.0)
        area = np.pi * (hi ** 2 - lo ** 2) * 1e6
        out.append([np.sum(m[s]) / area, np.std(v[s, 2]), np.mean(vphi[s])])
    return np.array(out)


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        p0, p1 = profiles(snaps[0]), profiles(snaps[-1])
        ctx.info[name] = {"annuli_kpc": ANNULI, "t0": p0.tolist(), "t_end": p1.tolist()}
        for (lo, hi), a, b in zip(ANNULI, p0, p1):
            tag = f"{name} R={lo:g}-{hi:g} kpc"
            checks.append(Check(f"{tag}: Sigma_* ratio", b[0] / a[0], 1.0, kind="rel", tol=0.10,
                                source="isolated-galaxy-ics.md equilibrium test"))
            checks.append(Check(f"{tag}: sigma_z ratio", b[1] / a[1], 1.0, kind="rel", tol=0.20,
                                source="isolated-galaxy-ics.md equilibrium test"))
            checks.append(Check(f"{tag}: v_phi ratio", b[2] / a[2], 1.0, kind="rel", tol=0.05,
                                source="isolated-galaxy-ics.md equilibrium test"))
    return checks
