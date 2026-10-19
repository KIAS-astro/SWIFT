"""Wind-carrier launch rate: launched mass against eta_M times the star
formation, and its scatter, for N_c = 1, 8, 64 (X2).

The star-forming layer of star_formation/prfm_slab (1 x 1 x 0.5 kpc periodic
box, 40 x 40 x 20 particles, Sigma_g = 200 Msun/pc^2, m_g = 6250 Msun,
every particle star-forming, t_dep = 263 Myr) with Wind:mode = carrier at
constant loadings eta_M = 0.2, eta_E = 0.1. Each step a star-forming
particle launches floor(p_c) carriers plus one with probability
p_c - floor(p_c), p_c = eta_M mdot_* dt / m_c, m_c = m_g / N_c. Deposit is
pushed beyond the run (d_rec = 1e3 kpc, t_rec = 1e4 Myr), so every carrier
launched is in the last snapshot with its launch time. Run for 50 Myr
(SWIFT_REG_CARRIER_TEND_MYR), once per N_c (SWIFT_REG_CARRIER_NC, default
"1,8,64").

The expected launched mass in a 10 Myr bin is E_b = eta_M sum SFR dt over
the active particles of the steps in the bin (SFR.txt column SFRdt_active,
the PRFM rate before the launch). The launched mass L_b is the mass of the
carriers with launch times in the bin. Each draw launches mass with
variance m_c^2 f (1 - f), f = p_c - floor(p_c) <= 0.06 here, so
var(L_b) = <m_c> E_b to within 6 per cent.

Checks, for each N_c:
  1. total launched mass / eta_M sum SFR dt over the run within 3 sigma,
     sigma = sqrt(<m_c> / E_total);
  2. chi^2 = sum_b (L_b - E_b)^2 / (<m_c> E_b) over the five bins within
     the 0.1-99.9 per cent range of chi^2 with 5 degrees of freedom
     (0.21-20.5): the scatter of the binned launched mass is the sampling
     scatter, which falls as N_c^-1/2;
  3. no carrier creation failed (spare star slots, Scheduler:
     cell_extra_sparts 2000);
and across N_c: the rms of L_b / E_b - 1 over the bins, divided by the
predicted sqrt(<m_c> / E_b), is recorded with the chi^2 (ctx.info).
Recorded: number of carriers, the rms and the predicted scatter per N_c.
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _common

DESCRIPTION = "wind carriers: launched mass = eta_M SFR dt, scatter falls as N_c^-1/2 (N_c = 1, 8, 64)"
TIER = "long"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature",
         "--feedback"]

SIGMA = 200.0  # Msun / pc^2
ETA_M, ETA_E = 0.2, 0.1
T_END_MYR = float(os.environ.get("SWIFT_REG_CARRIER_TEND_MYR", "50"))
N_CS = [int(x) for x in os.environ.get("SWIFT_REG_CARRIER_NC", "1,8,64").split(",")]
BIN_MYR = 10.0
CHI2_5DOF = (0.21, 20.5)  # 0.1 and 99.9 per cent points


def run(ctx):
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    myr = _common.MYR
    for n_c in N_CS:
        name = f"nc{n_c}"
        os.makedirs(ctx.path(name), exist_ok=True)
        Sigma = slab.write_layer(ctx.path(name, "ics.hdf5"), SIGMA)
        t_dep = slab.t_dep_layer(Sigma) / units.UNIT_TIME
        p = params.eagle_base("ics.hdf5", T_END_MYR * myr, ctx.data_dir,
                              T_END_MYR * myr, dt_max=0.5 * myr,
                              cooling=params.PRFM_COOLING)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
        p["SPH"]["h_max"] = 2.0 * 1.2348 * slab.LX / slab.NX
        p["Scheduler"].update({"cell_extra_sparts": 2000, "cell_extra_gparts": 2000})
        p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
        p["PRFMStarFormation"] = {
            "calibration": "OK22", "threshold": "measured",
            "threshold_n_H_p_cm3": 0.13, "use_dark_matter": 0,
            "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
            "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
            "map_half_width_kpc": 0.5 * slab.LX}
        p["Wind"] = _common.carrier_params(1e3, 1e4, n_c=n_c, model="constant",
                                           eta_M=ETA_M, eta_E=ETA_E,
                                           launch_direction=[0.0, 0.0, 1.0])
        ctx.info[name] = {"t_dep_Myr": t_dep * units.UNIT_TIME / units.MYR}
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=6 * 3600)


def expected_per_bin(sfr_file, edges):
    """eta_M sum SFRdt_active per bin of time (internal units)."""
    d = np.loadtxt(sfr_file, comments="#")
    t, sfrdt = d[:, 1], d[:, 6]
    idx = np.digitize(t, edges) - 1
    out = np.zeros(len(edges) - 1)
    for b in range(len(out)):
        out[b] = ETA_M * sfrdt[idx == b].sum()
    return out


def analyze(ctx):
    checks = []
    myr = _common.MYR
    edges = np.arange(0.0, T_END_MYR + 0.5 * BIN_MYR, BIN_MYR) * myr
    summary = {}
    for n_c in N_CS:
        name = f"nc{n_c}"
        last = snapshots(ctx.path(name))[-1]
        flag = last.read(4, "WindCarrier", cgs=False) > 0
        t_l = last.read(4, "WindCarrierLaunchTimes", cgs=False)[flag]
        m = last.read(4, "Masses", cgs=False)[flag]
        L = np.histogram(t_l, bins=edges, weights=m)[0]
        E = expected_per_bin(ctx.path(name, "SFR.txt"), edges)
        m_mean = m.mean() if m.size else 0.0
        sig_tot = np.sqrt(m_mean / max(E.sum(), 1e-30))
        checks.append(Check(f"N_c = {n_c}: launched / eta_M sum SFR dt", L.sum() / E.sum(),
                            1.0, kind="sigma", tol=3.0, sigma=sig_tot,
                            source=f"{flag.sum()} carriers; sigma = sqrt(<m_c> / E)"))
        chi2 = float(np.sum((L - E) ** 2 / (m_mean * E)))
        checks.append(Check(f"N_c = {n_c}: chi^2 of 10 Myr bins (5 dof) >= 0.1%",
                            chi2, CHI2_5DOF[0], kind="min",
                            source="var(L_b) = <m_c> E_b"))
        checks.append(Check(f"N_c = {n_c}: chi^2 of 10 Myr bins (5 dof) <= 99.9%",
                            chi2, CHI2_5DOF[1], kind="max", source="as above"))
        tail = _common.carrier_tail(ctx.path(name, f"{name}.log"))
        checks.append(Check(f"N_c = {n_c}: failed creations", tail["n_failed"] if tail else -1,
                            0, kind="equal", source="cell_extra_sparts = 2000"))
        rms = float(np.sqrt(np.mean((L / E - 1) ** 2)))
        pred = float(np.sqrt(np.mean(m_mean / E)))
        summary[name] = {"carriers": int(flag.sum()), "m_c_Msun": m_mean * units.UNIT_MASS
                         / units.SOLAR_MASS, "ratio_total": float(L.sum() / E.sum()),
                         "sigma_total": float(sig_tot), "chi2": chi2,
                         "rms_bin": rms, "predicted_rms_bin": pred,
                         "L_over_E_bins": (L / E).tolist()}
    ctx.info.update(summary)
    return checks
