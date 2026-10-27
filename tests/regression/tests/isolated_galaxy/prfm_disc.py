"""Milky Way-mass isolated disc with PRFM-int star formation (long tier).

Phase B4 of the PRFM design (ai-notes docs-code/SWIFT-star-formation/
prfm-design.md): the H12 disc with PRFM-int star formation and the TIGRESS
effective EoS as a floor, EAGLE cooling, and no feedback (D2: with PRFM the
feedback is a decoupled wind, which does not exist yet).

ICs: the H12 disc of Chaikin et al. (2023) at the two reference resolutions,
H12_m1.6e6 (gas particle mass 1.6e6 Msun, softening 0.5 kpc) and H12_m1.3e7
(1.3e7 Msun, 1.0 kpc), made by swift-ism-model tests/isolated_galaxy/
make_ics.py and read from <data>/ics/. Hernquist halo with M200 = 1.37e12 Msun,
c = 9, disc fraction 0.04, bulge fraction 0.014, as isolated_galaxy/
fkin_series in SWIFT-tk and the SWIFT IsolatedGalaxy examples. The H12 ICs
have no bulge particles: the bulge mass, 0.014 M200 = 1.9e10 Msun (1 per cent
of the halo mass), is subtracted from the Hernquist halo and carried by no
particle. SWIFT_REG_ISOGAL_BULGE overrides the bulge fraction (0 puts that mass
back into the halo). The PRFM face-on map is centred on the box centre (the
potential's centre) with the disc normal along z; rho_d comes from the same
Hernquist halo.

The runs below up to ok22_vol use J26's face-on cylinder: 2D cubic spline of
support 500 pc, the particle's own mass included (cylinder_kernel = spline,
cylinder_radius_kpc = 0.5, exclude_own_mass = 0), as before those options
existed. The runs from ok22_fix use the module defaults: a top hat of 1 kpc^2
(radius 1 / sqrt(pi) kpc) without the particle's own mass.

Runs (SWIFT_REG_PRFM_RUNS, comma-separated; default all):
  ok22      OK22 calibration, threshold n_H >= 0.13 cm^-3 on the measured density
  ok22_eq   OK22, threshold on the equilibrium density Sigma_g / (2 H_g 1.4 m_H)
            (and a measured n_H >= 0.01 cm^-3, the module default)
  k24       K24 calibration for star formation and floor
  k24_lowz  K24 with gas and stars at Z' = 0.1 (Z = 0.1 x 0.014): every
            EAGLEChemistry init_abundance_* metal entry scaled by the same
            factor, and H and He raised in proportion so that the mass
            fractions still sum to 1
  ok22_kc        OK22 with Sigma_g from the kernel column rho_SPH H_k / 1.4
                 (surface_density_estimator = kernel_column), uncorrected;
                 Sigma_*, sigma_*z from the map as in ok22
  ok22_kc_const  the same divided by 0.65 (kernel_column_correction = constant)
  ok22_kc_eq     the same corrected for the equilibrium thickness
                 (kernel_column_correction = equilibrium)
  ok22_vol  OK22, rate from PRFM-vol (formulation = vol, J26 Secs. 2.4-2.5):
            P_eff of the EoS at the measured density, t_dyn of J26 eq. 26 with
            rho_* = Sigma_* / (2 H_*), H_* / H_g = sigma_*z / sigma_eff from
            the map, R_f = 2; the PRFM-int state is still written
  ok22_fix    OK22, Sigma_g, Sigma_*, sigma_*z from the face-on top hat of
              1 kpc^2 without the particle's own mass
  ok22_kc116  OK22, Sigma_g from the kernel column, uncorrected, t_dep divided
              by 1.16 (kernel_column_rate_factor; ai-notes prfm-method-spec.md
              Sec. 4.2); Sigma_*, sigma_*z from the face-on top hat of 1 kpc^2
Resolution: SWIFT_REG_ISOGAL_RES = H12_m1.6e6 (default) or H12_m1.3e7, or
H12_m2e5 (2e5 Msun, softening 0.25 kpc; make_ics.py M5_disk.hdf5 2), the
high-resolution reference for the surface-density estimators.
Length: SWIFT_REG_PRFM_TEND_MYR (default 1000), snapshots every 10 Myr.
SPH h_max: SWIFT_REG_ISOGAL_HMAX [kpc] (default 10); particles in outflows
and in a hot CGM need h > 10 kpc.

Recorded: star formation history from birth times. Checks: each run reached
its end time; stars formed; the mass of stars formed equals the SFR summed
over gas particles and integrated over the snapshots (trapezoid rule),
within 4 Poisson sigma of the number of conversions plus the trapezoid error
estimated as the difference from the left-point sum; and the PRFM state
written for the star-forming particles of the last snapshot satisfies its
defining equations (state_checks).
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "isolated MW-mass disc, PRFM-int SF + TIGRESS EoS floor, no feedback"
TIER = "long"
BUILDS = {"prfm": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=hernquist"]}

RES = os.environ.get("SWIFT_REG_ISOGAL_RES", "H12_m1.6e6")
SOFTENING = {"H12_m2e5": 0.25, "H12_m1.6e6": 0.5, "H12_m1.3e7": 1.0}[RES]
IC_NAME = f"{RES}_disk.hdf5"
MYR_INTERNAL = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_PRFM_TEND_MYR", "1000"))
T_END = T_END_MYR * MYR_INTERNAL
DT_SNAP = 10.0 * MYR_INTERNAL
H_MAX = float(os.environ.get("SWIFT_REG_ISOGAL_HMAX", "10"))
BULGE_FRACTION = float(os.environ.get("SWIFT_REG_ISOGAL_BULGE", "0.014"))
# K24 Sec. 2.3 (Asplund et al. 2009), the PRFM modules' default
Z_SUN = 0.014
ALL_RUNS = {
    "ok22": ({"calibration": "OK22", "threshold": "measured"}, "OK22", None),
    "ok22_eq": ({"calibration": "OK22", "threshold": "equilibrium"}, "OK22", None),
    "k24": ({"calibration": "K24", "threshold": "measured"}, "K24", None),
    "k24_lowz": ({"calibration": "K24", "threshold": "measured"}, "K24", 0.1),
    "ok22_kc": ({"calibration": "OK22", "threshold": "measured",
                 "surface_density_estimator": "kernel_column"}, "OK22", None),
    "ok22_kc_const": ({"calibration": "OK22", "threshold": "measured",
                       "surface_density_estimator": "kernel_column",
                       "kernel_column_correction": "constant"}, "OK22", None),
    "ok22_vol": ({"calibration": "OK22", "threshold": "measured",
                  "formulation": "vol"}, "OK22", None),
    "ok22_kc_eq": ({"calibration": "OK22", "threshold": "measured",
                    "surface_density_estimator": "kernel_column",
                    "kernel_column_correction": "equilibrium"}, "OK22", None),
}
J26_CYLINDER = {"cylinder_kernel": "spline", "cylinder_radius_kpc": 0.5,
                "exclude_own_mass": 0}
for _sf, _cal, _z in ALL_RUNS.values():
    _sf.update(J26_CYLINDER)
ALL_RUNS.update({
    "ok22_fix": ({"calibration": "OK22", "threshold": "measured"}, "OK22", None),
    "ok22_kc116": ({"calibration": "OK22", "threshold": "measured",
                    "surface_density_estimator": "kernel_column",
                    "kernel_column_rate_factor": 1.16}, "OK22", None),
})
_only = os.environ.get("SWIFT_REG_PRFM_RUNS")
RUNS = {k: v for k, v in ALL_RUNS.items() if not _only or k in _only.split(",")}


def upsilon(W, cal, zprime):
    """Upsilon_tot in cm/s at weight W [erg/cm^3]: OK22 eq. 25c, K24 eq. 20."""
    w4 = W / (1e4 * units.BOLTZMANN_K)
    if cal == "K24":
        return 1650e5 * w4 ** -0.29 * np.maximum(zprime, 0.1) ** -0.27
    return 1028e5 * w4 ** -0.212


def sigma_eff(W, cal, zprime):
    """sigma_eff in cm/s at weight W: OK22 (floored at P_0, J26), K24 eq. 23."""
    w4 = W / (1e4 * units.BOLTZMANN_K)
    if cal == "K24":
        return 11.7e5 * w4 ** 0.12 * np.maximum(zprime, 0.1) ** 0.03
    return 12e5 * np.maximum(w4, 1.0) ** 0.22


def weight(Sg, Ss, rho_d, sig, sig_sz, c_d=16 * np.pi / 3):
    """J26 eq. 31 [erg/cm^3], cgs arrays; c_d = 16 pi / 3, the module default."""
    G = units.G_NEWTON
    Seff = Sg + 2 * Ss / (1 + sig_sz / sig)
    x = c_d * G * rho_d * sig ** 2 / (np.pi * G * Seff) ** 2
    return 0.25 * np.pi * G * Sg * Seff * (1 + np.sqrt(1 + x))


def vol_density(t_dep, t_dyn, R_f=2.0):
    """Gas density [g/cm^3] that PRFM-vol (OK22, J26 EoS) used, from its t_dep / t_dyn.

    t_dep / t_dyn = Upsilon(P) / (R_f sigma_eff(P)) = (1028 / 12) x^-0.212 / R_f
    for x = P / P_0 <= 1 and (1028 / 12) x^-0.432 / R_f above; then
    P / k_B = 4.7e4 n_H^1.8, n_H = rho / (1.4 m_H).
    """
    r = R_f * t_dep / t_dyn / (1028.0 / 12.0)
    x = np.where(r >= 1, r ** (-1 / 0.212), r ** (-1 / 0.432))
    n_H = (x * 1e4 / 4.7e4) ** (1 / 1.8)
    return n_H * 1.4 * units.PROTON_MASS


def vol_t_dyn(rho_g, Ss, rho_d, sig_sz):
    """J26 eq. 26 with rho_* = Sigma_* / (2 H_*), H_* / H_g = sigma_*z / sigma_eff [s]."""
    G = units.G_NEWTON
    P = 4.7e4 * units.BOLTZMANN_K * (rho_g / (1.4 * units.PROTON_MASS)) ** 1.8
    sig = sigma_eff(P, "OK22", 1.0)
    A = 2 * np.pi * G * rho_g + 4 * np.pi / 3 * G * rho_d
    B = 4 * np.pi * G * Ss / (sig + sig_sz)
    return 8 / (B + np.sqrt(B * B + 16 * A))


def ic_file(data_dir):
    path = os.path.join(data_dir, "ics", IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}: make it with swift-ism-model tests/isolated_galaxy/make_ics.py")
    return path


def run(ctx):
    for t in ("coolingtables", "photometry"):
        ctx.table(t)
    ics = ic_file(ctx.data_dir)
    for name, (sf, floor_cal, zprime) in RUNS.items():
        os.makedirs(ctx.path(name))
        shutil.copy(ics, ctx.path(name, IC_NAME))
        p = params.eagle_base(IC_NAME, T_END, ctx.data_dir, DT_SNAP, dt_max=1e-2)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
        p["Snapshots"]["compression"] = 4
        p["Gravity"].update({"max_physical_DM_softening": SOFTENING,
                             "max_physical_baryon_softening": SOFTENING})
        p["SPH"].update({"h_min_ratio": 0.1, "h_max": H_MAX,
                         "minimal_temperature": 100.0})
        if zprime is not None:
            p["EAGLEChemistry"] = params.eagle_abundances(zprime * Z_SUN)
        p["Scheduler"] = {"max_top_level_cells": 16, "tasks_per_cell": 200}
        p["HernquistPotential"] = {
            "useabspos": 0, "position": [0.0, 0.0, 0.0], "idealizeddisk": 1,
            "M200": 137.0, "h": 0.704, "concentration": 9.0, "diskfraction": 0.040,
            "bulgefraction": BULGE_FRACTION, "timestep_mult": 0.01,
            "epsilon": SOFTENING}
        p["PRFMEntropyFloor"] = {"calibration": floor_cal,
                                 "density_threshold_H_p_cm3": 0.13}
        p["PRFMStarFormation"] = dict(
            sf, threshold_n_H_p_cm3=0.13, use_dark_matter=1,
            disc_centre=[0.0, 0.0, 0.0], disc_centre_is_absolute=0, pixel_kpc=0.1,
            half_height_kpc=1.0, map_half_width_kpc=25.0)
        ctx.swift(p, ["--hydro", "--self-gravity", "--external-gravity", "--stars",
                      "--star-formation", "--cooling", "--limiter", "--sync"],
                  subdir=name, name=name, timeout=12 * 3600)


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        birth = last.read(4, "BirthTimes") / units.MYR
        m_init = last.read(4, "InitialMasses")
        new = birth > 0
        edges = np.arange(0, T_END_MYR + 1e-6, 10.0)
        sfr_hist = np.histogram(birth[new], edges, weights=m_init[new])[0] / (
            units.SOLAR_MASS * 10e6)

        t = np.array([s.time_cgs for s in snaps])
        sfr_tot = np.array([np.clip(s.read(0, "StarFormationRates"), 0, None).sum()
                            for s in snaps])
        # The output at t = 0 precedes the first SFR computation
        sfr_tot[0] = sfr_tot[1]
        expected = np.sum(0.5 * (sfr_tot[1:] + sfr_tot[:-1]) * np.diff(t))
        left = np.sum(sfr_tot[:-1] * np.diff(t))
        formed = m_init[new].sum()
        m_g = np.median(last.read(0, "Masses"))
        sigma = np.hypot(np.sqrt(max(new.sum(), 1)) * m_g, expected - left)

        ctx.info[name] = {
            "t_Myr": (0.5 * (edges[1:] + edges[:-1])).tolist(),
            "sfr_Msun_yr": sfr_hist.tolist(),
            "stars_formed": int(new.sum()),
            "sfr_snapshots_Msun_yr": (sfr_tot * units.YEAR / units.SOLAR_MASS).tolist(),
        }
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-2, source="run completed"))
        checks.append(Check(f"{name}: stars formed", new.sum(), 1, kind="min",
                            source="PRFM-int rate in a gas disc"))
        checks.append(Check(f"{name}: mass formed / integrated SFR",
                            formed / units.SOLAR_MASS, expected / units.SOLAR_MASS,
                            kind="sigma", tol=4.0, sigma=sigma / units.SOLAR_MASS,
                            source="sum of gas SFR over snapshots, trapezoid rule; "
                                   "Poisson sigma of the conversions plus the "
                                   "trapezoid-vs-left-point difference"))
        checks += state_checks(name, last)
    return checks


def state_checks(name, snap):
    """The PRFM state written for star-forming particles, against its definitions.

    All fields of a particle are written at the same step as its SFR.
    Identities (float32 fields, relative rounding 6e-8 each; tolerance 1e-5):
    W = J26 eq. 31 of the written Sigma_g, Sigma_*, rho_d, sigma_eff,
    sigma_*z; SFR = m_g / t_dep; rho_eq = W / sigma_eff^2,
    H_g = Sigma_g / (2 rho_eq); for PRFM-int t_dep = Upsilon(W) Sigma_g / W,
    divided by kernel_column_rate_factor in ok22_kc116, and
    t_dyn = 2 H_g / sigma_eff. For PRFM-vol (ok22_vol) the density the
    rate used is recovered from t_dep / t_dyn = Upsilon(P) / (R_f sigma(P))
    and the EoS, and t_dyn must equal J26 eq. 26 at that density with the
    written Sigma_*, sigma_*z, rho_d (tolerance 1e-5: the inversion raises
    t_dep / t_dyn to the power -1 / (1.8 x 0.432), which multiplies its
    rounding by 1.3, and t_dyn goes as rho^-1/2; measured 1.2e-7); the
    recovered density must match the written Densities to 10 per cent in the
    median, since inactive particles have drifted since their last active
    step. sigma_eff = sigma_eff(W) holds only to the
    convergence of the 5 fixed-point iterations: each one multiplies the
    error by at most d ln sigma / d ln W = beta <= 0.22, so 5 leave
    0.22^5 = 5e-4 of the initial error, which is at most a factor 3
    (10 km/s against 33 km/s); tolerance 2e-3. For ok22_eq every
    star-forming particle has n_H,eq >= 0.13 cm^-3.
    """
    sf_cal = ALL_RUNS[name][0]["calibration"]
    rate_factor = ALL_RUNS[name][0].get("kernel_column_rate_factor", 1.0)
    sfr = snap.read(0, "StarFormationRates")
    sf = sfr > 0
    m = snap.read(0, "Masses")[sf]
    zprime = snap.read(0, "SmoothedMetalMassFractions")[sf] / Z_SUN
    Sg = snap.read(0, "PRFMGasSurfaceDensities")[sf]
    W = snap.read(0, "PRFMWeights")[sf]
    sig = snap.read(0, "PRFMEffectiveDispersions")[sf]
    rho_eq = snap.read(0, "PRFMEquilibriumDensities")[sf]
    H = snap.read(0, "PRFMEquilibriumScaleHeights")[sf]
    Ss = snap.read(0, "PRFMStellarSurfaceDensities")[sf]
    sig_sz = snap.read(0, "PRFMStellarVerticalDispersions")[sf]
    rho_d = snap.read(0, "PRFMDarkMatterDensities")[sf]
    t_dep = snap.read(0, "PRFMDepletionTimes")[sf]
    t_dyn = snap.read(0, "PRFMDynamicalTimes")[sf]
    src = f"{name}: {sf.sum()} star-forming particles, last snapshot"
    tests = [
        ("W / J26 eq. 31", W, weight(Sg, Ss, rho_d, sig, sig_sz), 1e-5),
        ("SFR / (m_g / t_dep)", sfr[sf], m / t_dep, 1e-5),
        ("rho_eq / (W / sigma_eff^2)", rho_eq, W / sig ** 2, 1e-5),
        ("H_g / (Sigma_g / 2 rho_eq)", H, Sg / (2 * rho_eq), 1e-5),
        ("sigma_eff / sigma_eff(W)", sig, sigma_eff(W, sf_cal, zprime), 2e-3)]
    vol = ALL_RUNS[name][0].get("formulation") == "vol"
    if vol:
        rho_used = vol_density(t_dep, t_dyn)
        tests.append(("t_dyn / J26 eq. 26", t_dyn,
                      vol_t_dyn(rho_used, Ss, rho_d, sig_sz), 1e-5))
    else:
        tests += [
            ("t_dep / (Upsilon Sigma_g / W / rate factor)", t_dep,
             upsilon(W, sf_cal, zprime) * Sg / W / rate_factor, 1e-5),
            ("t_dyn / (2 H_g / sigma_eff)", t_dyn, 2 * H / sig, 1e-5)]
    out = []
    if vol:
        rho = snap.read(0, "Densities")[sf]
        out.append(Check(f"{name}: median |rho used / Densities - 1|",
                         np.median(np.abs(rho_used / rho - 1)), 0.1, kind="max",
                         source=src + "; drift since the last active step"))
    for label, got, want, tol in tests:
        out.append(Check(f"{name}: max |{label} - 1|",
                         np.abs(got / want - 1).max(), tol, kind="max",
                         source=src))
    if ALL_RUNS[name][0]["threshold"] == "equilibrium":
        n_eq = rho_eq / (1.4 * units.PROTON_MASS)
        out.append(Check(f"{name}: min n_H,eq of star-forming gas [cm^-3]",
                         n_eq.min(), 0.13 * (1 - 1e-6), kind="min",
                         source=src + "; equilibrium threshold"))
    return out
