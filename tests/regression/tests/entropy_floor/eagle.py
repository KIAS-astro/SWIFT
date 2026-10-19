"""EAGLE entropy floor: cooling stops at the floor pressure, never below it.

The floor (src/entropy_floor/EAGLE/entropy_floor.c) is the larger of
  Jeans floor  P_J (rho/rho_J)^(4/3), above n_H = 0.1 cm^-3,
  Cool floor   P_C (rho/rho_C),       above n_H = 1e-5 cm^-3,
with P = rho k T_norm / (mu_neutral m_p) at each threshold, T_norm = 8000 K
(EAGLE reference model, Schaye et al. 2015, Sec. 4.3). It is imposed in the
cooling step only (cooling/EAGLE/cooling.c), as u = max(u_cooled, u_floor).
In a non-cosmological run the over-density conditions always hold
(critical density 0).

Uniform boxes with EAGLE cooling at z = 0 (Wiersma et al. 2009 tables, solar
metallicity) for 5 Myr:
  n_H = 1, 10, 100 cm^-3 from 100 K: radiative equilibrium lies below the
      Jeans floor (1.7e4, 3.7e4, 8.0e4 K), so the gas must sit on the floor;
  n_H = 10 cm^-3 from 1e6 K: cools within ~0.1 Myr and must stop on the floor;
  n_H = 0.01 cm^-3 from 100 K: photoheated towards ~1e4 K, so it must be at or
      above its 8000 K floor.
The check uses pressures from the snapshot, so no mean molecular weight has
to be assumed for a temperature.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "EAGLE entropy floor: dense gas cools onto the floor pressure and not below"
TIER = "quick"
BUILDS = {"eagle": ["--with-subgrid=EAGLE"]}

N = 16
BOX = 0.1  # kpc
T_END = 5e-3  # internal units, 4.9 Myr
CASES = {
    "n1_cold": (1.0, 1e2, "on"),
    "n10_cold": (10.0, 1e2, "on"),
    "n100_cold": (100.0, 1e2, "on"),
    "n10_hot": (10.0, 1e6, "on"),
    "n0.01_cold": (0.01, 1e2, "above"),
}


def floor_pressure(rho, floor):
    """Floor pressure in cgs for physical density rho in g/cm^3."""
    mu = units.MU_NEUTRAL
    out = np.zeros_like(rho)
    for kind in ("Jeans", "Cool"):
        n_th = floor[f"{kind}_density_threshold_H_p_cm3"]
        rho_th = units.rho_from_nH(n_th)
        P_th = rho_th * units.BOLTZMANN_K * floor[f"{kind}_temperature_norm_K"] / (
            mu * units.PROTON_MASS)
        gamma = floor[f"{kind}_gamma_effective"]
        P = np.where(rho >= rho_th, P_th * (rho / rho_th) ** gamma, 0.0)
        out = np.maximum(out, P)
    return out


def run(ctx):
    ctx.table("coolingtables")
    for name, (nH, T, _) in CASES.items():
        gas, _ = ics.uniform_gas(N, BOX, nH, T)

        os.makedirs(ctx.path(name))
        ics.write(ctx.path(name, "ics.hdf5"), BOX, gas=gas)
        p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END, dt_max=1e-4)
        ctx.info["floor"] = p["EAGLEEntropyFloor"]
        ctx.swift(p, ["--hydro", "--cooling"], subdir=name, name=name)


def analyze(ctx):
    floor = ctx.info["floor"]
    checks = []
    for name, (nH, T, expect) in CASES.items():
        last = snapshots(ctx.path(name))[-1]
        rho = last.read(0, "Densities")
        P = last.read(0, "Pressures")
        ratio = P / floor_pressure(rho, floor)
        src = (f"n_H = {nH} cm^-3, start T = {T:g} K; floor from EAGLEEntropyFloor "
               f"with mu_neutral = {units.MU_NEUTRAL:.4f}")
        checks.append(Check(f"{name}: min P / P_floor", ratio.min(), 1.0 - 1e-3,
                            kind="min", source=src + "; never below the floor"))
        if expect == "on":
            checks.append(Check(f"{name}: max P / P_floor", ratio.max(), 1.0 + 1e-2,
                                kind="max",
                                source=src + "; cooling equilibrium lies below the floor"))
        ctx.info[f"{name}_P_over_Pfloor"] = [ratio.min(), ratio.max()]
    return checks
