"""GIZMO: the EAGLE entropy floor holds gas at or above the floor.

Uniform gas at rest, n_H = 1 cm^-3, T = 5000 K, hydro only (no cooling;
--temperature because the EAGLE tracers need it).
The EAGLE floor (Schaye et al. 2015, the reference parameters of
params.eagle_base: Jeans floor 8000 K at n_H = 0.1 cm^-3 with slope
gamma_eff - 1 = 1/3) gives T_floor = 8000 K (n_H / 0.1)^(1/3) = 17,240 K at
n_H = 1, in SWIFT's convention u_floor = k T_floor / ((gamma - 1) mu_neutral m_p)
(src/entropy_floor/EAGLE/entropy_floor.c). SPHENIX applies it in
hydro_predict_extra and hydro_kick_extra, so the gas sits on the floor from
the first output.

Before the fix GIZMO never applied the floor (hydro.h carried
"MATTHIEU: Apply the entropy floor here" placeholders): measured 2026-10-20,
gizmo-mfm, min u / u_floor = 0.290 at t = 0 and at 1 Myr (the gas stays at
5000 K). After: 1.000 at both, as SPHENIX (SWIFT_REG_HYDRO=sphenix).

Checks (t = 0 and t = 1 Myr): min over particles of u / u_floor(rho) >= 0.99.
The 1 per cent allows float rounding of the floor; any particle left below
the floor shows up as the ratio 5000 / 17,240 = 0.29.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "GIZMO: EAGLE entropy floor applied at t = 0 and during the run"
TIER = "quick"
HYDRO = os.environ.get("SWIFT_REG_HYDRO", "gizmo-mfm")
BUILDS = {"eagle": ["--with-subgrid=EAGLE"]
          + ([] if HYDRO == "sphenix" else
             [f"--with-hydro={HYDRO}", "--with-riemann-solver=hllc"])}

N = 16
BOX = 0.2  # kpc
NH = 1.0
T0 = 5000.0
T_END = 1.0227e-3  # 1 Myr


def run(ctx):
    gas, _ = ics.uniform_gas(N, BOX, NH, T0)
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas)
    ctx.table("coolingtables")
    ctx.table("yieldtables")
    ctx.table("photometry")
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END, dt_max=1e-4)
    ctx.swift(p, ["--hydro", "--temperature"], timeout=600)


def u_floor(rho_cgs, fl):
    nH = rho_cgs * units.HYDROGEN_FRACTION / units.PROTON_MASS
    T_jeans = np.where(nH >= fl["Jeans_density_threshold_H_p_cm3"],
                       fl["Jeans_temperature_norm_K"]
                       * (nH / fl["Jeans_density_threshold_H_p_cm3"])
                       ** (fl["Jeans_gamma_effective"] - 1.0), 0.0)
    T_cool = np.where(nH >= fl["Cool_density_threshold_H_p_cm3"],
                      fl["Cool_temperature_norm_K"]
                      * (nH / fl["Cool_density_threshold_H_p_cm3"])
                      ** (fl["Cool_gamma_effective"] - 1.0), 0.0)
    return units.u_from_T(np.maximum(T_jeans, T_cool), units.MU_NEUTRAL)


def analyze(ctx):
    fl = params.eagle_base("x", 1, "x", 1)["EAGLEEntropyFloor"]
    snaps = snapshots(ctx.run_dir)
    out = []
    for s, label in [(snaps[0], "t = 0"), (snaps[-1], "t = 1 Myr")]:
        rho = s.read(0, "Densities")
        u = s.read(0, "InternalEnergies")
        r = float(np.min(u / u_floor(rho, fl)))
        ctx.info[f"min_u_over_floor {label}"] = r
        out.append(Check(f"min u / u_floor at {label}", r, 0.99, kind="min",
                         source="EAGLE Jeans floor, entropy_floor/EAGLE/entropy_floor.c"))
    return out
