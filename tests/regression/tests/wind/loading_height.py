"""Loadings taken at the recoupling height (Wind:loading_height = recoupling;
wind-design.md Sec. 9.1 and 9.4): validation against the TIGRESS outflow
tables of Kim et al. (2020b, K20b).

The uniform star-forming layer of wind/launch_rate (1 x 1 x 0.5 kpc,
Sigma_g = 200 Msun/pc^2, PRFM-int OK22) with the TIGRESS loadings
(enthalpy convention), loading_height = recoupling and d_rec = 2 kpc, so
every launch uses the K20b hot-phase fits at |z| = 1 kpc:
    eta_M = 10^-1.185 Sigma_SFR^-0.100,  eta_E = 10^-1.381 Sigma_SFR^0.013,
Sigma_SFR in Msun kpc^-2 yr^-1, clipped to [1e-4, 1]. Run for 2 per cent
of t_dep.

Checks:
  1. per launch, the eta_M and eta_E recorded at launch against the 1 kpc
     fits at the recorded Sigma_SFR, relative 1e-5 (float storage);
  2. launches against sum over stars of eta_M: N_w - <eta_M> N_* within
     3 sigma, sigma^2 = <eta_M> (1 + <eta_M>) N_* (wind/launch_rate);
  3. launched Bernoulli energy, sum of v^2/2 + gamma u over launches,
     against the sum of eta_E u_* / eta_M (enthalpy convention, WD3),
     relative 1e-3; per particle the gas velocity before the kick changes
     it by up to 0.3 per cent, but with random signs this averages out;
  4. validation of the fits against the published reference: at each of
     the seven K20b models (Sigma_SFR,40 and the time-averaged hot-phase
     loadings at |z| = 500 pc and 1 kpc from the K20b Zenodo table,
     doi:10.5281/zenodo.3872049, tables/table-mean.ecsv), the code's
     fits within 0.3 dex of the measured value. 0.3 dex is 2.3 times the
     rms scatter of the fits (0.10-0.16 dex) and 1.3-3 times their largest
     residual (0.13-0.23 dex);
  5. the factor applied: the recorded eta at launch over the K20b Table 5
     value at |z| = H at the same Sigma_SFR lies in 0.3-0.8 for eta_M and
     0.2-0.6 for eta_E (Sec. 9.1: 0.54 and 0.37 at Sigma_SFR = 1e-2).
"""

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _common

DESCRIPTION = "loadings at the recoupling height: K20b 1 kpc fits, against the K20b tables"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

SIGMA = 200.0  # Msun / pc^2
FRACTION = 0.02
D_REC = 2.0  # kpc
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g
GAMMA = 5.0 / 3.0

# log10 eta = a + b log10 Sigma_SFR, hot phase (wind-design.md Sec. 9.1)
FIT = {"H": {"M": (-0.86, -0.07), "E": (-0.70, 0.14)},
       "500": {"M": (-0.984, -0.145), "E": (-0.953, -0.006)},
       "1000": {"M": (-1.185, -0.100), "E": (-1.381, 0.013)}}

# K20b Zenodo table-mean.ecsv, phase hot: models R2, R4, R8, R16, LGR2,
# LGR4, LGR8
K20B_SFR40 = [1.128, 0.1247, 0.005137, 7.964e-05, 0.4871, 0.08968, 0.003209]
K20B = {"500": {"M": [0.096, 0.1912, 0.1838, 0.3852, 0.08772, 0.1687, 0.2721],
                "E": [0.1542, 0.1425, 0.06725, 0.177, 0.09724, 0.1031, 0.09484]},
        "1000": {"M": [0.05556, 0.114, 0.1381, 0.1213, 0.05365, 0.07496, 0.1542],
                 "E": [0.05622, 0.05371, 0.03222, 0.04383, 0.03746, 0.02518, 0.03879]}}


def fit(height, q, sigma_sfr):
    a, b = FIT[height][q]
    s = np.clip(sigma_sfr, 1e-4, 1.0)
    return 10.0 ** (a + b * np.log10(s))


def run(ctx):
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    Sigma = slab.write_layer(ctx.path("ics.hdf5"), SIGMA)
    t_dep = slab.t_dep_layer(Sigma) / units.UNIT_TIME
    t_end = FRACTION * t_dep
    p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, t_end,
                          dt_max=slab.DT_MAX * t_dep, cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
    p["SPH"]["h_max"] = 2.0 * 1.2348 * slab.LX / slab.NX
    p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
    p["PRFMStarFormation"] = {
        "calibration": "OK22", "threshold": "measured",
        "threshold_n_H_p_cm3": 0.13, "use_dark_matter": 0,
        "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
        "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
        "map_half_width_kpc": 0.5 * slab.LX}
    p["Wind"] = {"loading_model": "TIGRESS", "loading_height": "recoupling",
                 "launch_direction": [0.0, 0.0, 1.0],
                 "recoupling_distance_kpc": D_REC, "recoupling_time_Myr": 1e4}
    ctx.swift(p, FLAGS, name="wind", timeout=1800)


def analyze(ctx):
    checks = []
    last = snapshots(ctx.run_dir)[-1]
    n_star = int(last.npart[4])
    counts = last.read(0, "WindLaunchCounts", cgs=False)
    launched = counts > 0
    n_wind = int(counts.sum())
    sig = last.read(0, "WindLaunchSigmaSFR", cgs=False)[launched]
    eM = last.read(0, "WindLaunchMassLoadings", cgs=False)[launched]
    eE = last.read(0, "WindLaunchEnergyLoadings", cgs=False)[launched]
    checks.append(Check("launches", n_wind, 20, kind="min",
                        source="enough launches for the per-launch checks"))
    checks.append(Check("max |eta_M / eta_M,1kpc(Sigma_SFR) - 1|",
                        np.abs(eM / fit("1000", "M", sig) - 1).max(), 1e-5, kind="max",
                        source="K20b hot phase at 1 kpc (Sec. 9.1), d_rec = 2 kpc clamped"))
    checks.append(Check("max |eta_E / eta_E,1kpc(Sigma_SFR) - 1|",
                        np.abs(eE / fit("1000", "E", sig) - 1).max(), 1e-5, kind="max",
                        source="as above"))
    eta_mean = eM.mean()
    sigma = np.sqrt(eta_mean * (1 + eta_mean) * max(n_star, 1))
    checks.append(Check("launches / <eta_M> stars", n_wind, eta_mean * n_star,
                        kind="sigma", tol=3.0, sigma=sigma,
                        source=f"p_w = eta_M p_*, N_* = {n_star}"))
    u = last.read(0, "WindLaunchInternalEnergies")[launched]
    ek = last.read(0, "WindLaunchKineticEnergies")[launched]
    e_target = eE * U_STAR / eM
    checks.append(Check("sum (e_kin + gamma u) / sum eta_E u_* / eta_M",
                        np.sum(ek + GAMMA * u) / np.sum(e_target), 1.0, kind="rel",
                        tol=1e-3, source="enthalpy convention (WD3); the pre-launch gas "
                                         "velocity averages out over the random signs"))
    worst = 0.0
    for h in ("500", "1000"):
        for q in ("M", "E"):
            dex = np.abs(np.log10(fit(h, q, np.array(K20B_SFR40)) / np.array(K20B[h][q])))
            worst = max(worst, dex.max())
            checks.append(Check(f"fit vs K20b table, eta_{q} at {h} pc: max |dex|",
                                dex.max(), 0.3, kind="max",
                                source="K20b Zenodo table-mean.ecsv, 7 models"))
    fM = eM / fit("H", "M", sig)
    fE = eE / fit("H", "E", sig)
    checks.append(Check("f_M = eta_M / eta_M,H in [0.3, 0.8]: min", fM.min(), 0.3, kind="min",
                        source="Sec. 9.1"))
    checks.append(Check("f_M: max", fM.max(), 0.8, kind="max", source="Sec. 9.1"))
    checks.append(Check("f_E = eta_E / eta_E,H in [0.2, 0.6]: min", fE.min(), 0.2, kind="min",
                        source="Sec. 9.1"))
    checks.append(Check("f_E: max", fE.max(), 0.6, kind="max", source="Sec. 9.1"))
    ctx.info.update({"stars": n_star, "launches": n_wind,
                     "Sigma_SFR_median": float(np.median(sig)),
                     "eta_M_median": float(np.median(eM)), "eta_E_median": float(np.median(eE)),
                     "f_M_median": float(np.median(fM)), "f_E_median": float(np.median(fE)),
                     "fit_vs_table_worst_dex": worst})
    return checks
