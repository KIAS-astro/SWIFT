"""PRFM entropy floor: gas above n_th ends on the TIGRESS effective EoS.

The floor (src/entropy_floor/PRFM/entropy_floor.c) is
  P_eff / k_B = 4.7e4 n_H^1.8 K cm^-3          (OK22, as J26)
  log P_eff / k_B = 1.43 log n_H + 4.30         (OK22_fit: OK22 eq. 27)
  P_eff = rho sigma_eff(P_eff)^2                (K24, Z' clipped at 0.1)
with the K24 eq. 23 dispersion sigma_eff = 11.7 km/s (P/P_0)^0.12 Z'^0.03,
P_0 = 1e4 k_B K cm^-3, Z' = Z / 0.014. The expected K24 pressure is found
here by bisection on that implicit equation, not from the closed form the
code uses (3.03e4 n_H^1.316 Z'^0.079), so the check is independent of it.
with n_H = rho / (1.4 m_H), above n_th = 0.13 cm^-3 and nowhere below. Like
the EAGLE floor it is imposed in the cooling step, u = max(u_cooled, u_floor),
and in the hydro kick.

Uniform boxes with EAGLE cooling at z = 0 (Wiersma et al. 2009 tables, solar
metallicity) for 5 Myr:
  n_H = 1, 10, 100 cm^-3 from 100 K: radiative equilibrium (< 1e4 K) lies far
      below the floor (4.1e4, 2.6e5, 1.6e6 K for neutral gas), so the gas must
      sit on it;
  n_H = 10 cm^-3 from 1e7 K: cools onto the floor and stops there;
  n_H = 10 cm^-3 from 100 K with K24 at Z' = 0.0129 / 0.014 = 0.92 and at
      Z' = 0.1 (every EAGLE metal abundance scaled, H and He raised to keep
      the sum at 1);
  n_H = 10 cm^-3 from 100 K with eos_form OK22_fit;
and without cooling (hydro only, --temperature), where nothing changes u but the floor:
  n_H = 1 cm^-3 from 100 K: lifted onto the floor by the kick, J26 and
      OK22_fit;
  n_H = 0.1 cm^-3 from 100 K: below n_th, must stay at 100 K.
Here n_H is the J26 definition, so the boxes are written with
rho = 1.4 m_H n_H.

On the floor, hydrodynamics leaves density fluctuations of a few per cent in
the hot lattice (5 per cent at n_H = 100 after 5 Myr). A compressed particle
is held on the floor; an expanded one follows its adiabat, P ~ rho^(5/3),
which for a floor P ~ rho^1.8 lies above the floor by 0.13 times its density
deficit. For the softer OK22_fit floor, P ~ rho^1.43, the roles swap: an
expanded particle is held on the floor and a compressed one lies above it
by 0.24 times its density excess. So P / P_eff >= 1 is exact for every
particle, the median sits on the floor to 1e-3, and the maximum is allowed
1e-2, as in entropy_floor/eagle.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "PRFM entropy floor: gas above n_th = 0.13 ends on P_eff(n_H) (J26, OK22 fit, K24)"
TIER = "quick"
BUILDS = {"prfm": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=constant"]}

N = 16
BOX = 0.1  # kpc
T_END = 5e-3  # internal units, 4.9 Myr
Z_SUN = 0.014
Z = params.EAGLE_SOLAR_ABUNDANCES["init_abundance_metal"]
N_TH = 0.13
# name: (n_H [cm^-3], T0 [K], calibration, cooling, expected, Z); the
# calibration "OK22_fit" means calibration OK22 with eos_form OK22_fit
CASES = {
    "n1_cold": (1.0, 1e2, "OK22", True, "on", Z),
    "n10_cold": (10.0, 1e2, "OK22", True, "on", Z),
    "n100_cold": (100.0, 1e2, "OK22", True, "on", Z),
    "n10_hot": (10.0, 1e7, "OK22", True, "on", Z),
    "n10_cold_K24": (10.0, 1e2, "K24", True, "on", Z),
    "n10_cold_K24_lowZ": (10.0, 1e2, "K24", True, "on", 0.1 * Z_SUN),
    "n10_cold_OK22fit": (10.0, 1e2, "OK22_fit", True, "on", Z),
    "n1_nocool": (1.0, 1e2, "OK22", False, "on", Z),
    "n1_nocool_OK22fit": (1.0, 1e2, "OK22_fit", False, "on", Z),
    "n0.1_nocool": (0.1, 1e2, "OK22", False, "unchanged", Z),
}


def p_k24(rho_cgs, Zp):
    """Root of P = rho sigma_eff(P)^2 with K24 eq. 23, by bisection in log P."""
    P0 = 1e4 * units.BOLTZMANN_K
    lo = np.full(np.shape(rho_cgs), -30.0)
    hi = np.full(np.shape(rho_cgs), 0.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        P = 10.0 ** mid
        sigma = 11.7e5 * (P / P0) ** 0.12 * Zp ** 0.03
        above = P > rho_cgs * sigma ** 2
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 10.0 ** (0.5 * (lo + hi))


def p_eff(rho_cgs, cal, Zp=1.0):
    """Floor pressure in cgs at physical density rho."""
    n = rho_cgs / (1.4 * units.PROTON_MASS)
    Zp = max(Zp, 0.1)
    if cal == "K24":
        P = p_k24(rho_cgs, Zp)
    elif cal == "OK22_fit":
        P = 10.0 ** (1.43 * np.log10(n) + 4.30) * units.BOLTZMANN_K
    else:
        P = 4.7e4 * n ** 1.8 * units.BOLTZMANN_K
    return np.where(n >= N_TH, P, 0.0)


def run(ctx):
    ctx.table("coolingtables")
    for name, (nH, T, cal, cool, _, Zm) in CASES.items():
        # uniform_gas sets rho = n m_p / X_H; this gives rho = 1.4 m_p nH
        gas, _ = ics.uniform_gas(N, BOX, 1.4 * units.HYDROGEN_FRACTION * nH, T)
        os.makedirs(ctx.path(name))
        ics.write(ctx.path(name, "ics.hdf5"), BOX, gas=gas)
        p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END, dt_max=1e-4)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["EAGLEChemistry"] = params.eagle_abundances(Zm)
        p["PRFMEntropyFloor"] = {"calibration": cal, "density_threshold_H_p_cm3": N_TH,
                                 "solar_metallicity": Z_SUN}
        if cal == "OK22_fit":
            p["PRFMEntropyFloor"].update(calibration="OK22", eos_form="OK22_fit")
        # --temperature (required by the EAGLE tracers) applies no cooling
        flags = ["--hydro", "--cooling"] if cool else ["--hydro", "--temperature"]
        ctx.swift(p, flags, subdir=name, name=name)


def analyze(ctx):
    checks = []
    for name, (nH, T, cal, cool, expect, Zm) in CASES.items():
        last = snapshots(ctx.path(name))[-1]
        rho = last.read(0, "Densities")
        P = last.read(0, "Pressures")
        src = (f"n_H = {nH} cm^-3 (rho / 1.4 m_H), start T = {T:g} K, "
               f"{'cooling' if cool else 'no cooling'}; {cal}, Z' = {Zm / Z_SUN:.3g}")
        if expect == "on":
            ratio = P / p_eff(rho, cal, Zm / Z_SUN)
            checks.append(Check(f"{name}: min P / P_eff", ratio.min(), 1.0 - 1e-3,
                                kind="min", source=src + "; never below the floor"))
            checks.append(Check(f"{name}: median P / P_eff",
                                np.median(ratio), 1.0 + 1e-3, kind="max",
                                source=src + "; equilibrium lies below the floor"))
            checks.append(Check(f"{name}: max P / P_eff", ratio.max(), 1.0 + 1e-2,
                                kind="max",
                                source=src + "; rarefied particles may sit above "
                                "a floor stiffer than gamma = 5/3"))
            ctx.info[f"{name}_P_over_Peff"] = [ratio.min(), np.median(ratio),
                                               ratio.max()]
        else:
            P0 = rho * units.BOLTZMANN_K * T / (units.MU_NEUTRAL * units.PROTON_MASS)
            ratio = P / P0
            checks.append(Check(f"{name}: max |P / P_initial - 1|",
                                np.abs(ratio - 1).max(), 1e-3, kind="max",
                                source=src + "; below n_th there is no floor"))
    return checks
