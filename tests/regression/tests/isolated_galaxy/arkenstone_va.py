"""The isolated galaxy of Arkenstone I (Smith et al. 2024a, S24a) with PRFM star
formation, the TIGRESS effective EoS and the decoupled wind (long tier).

Test V-A of ai-notes docs-code/SWIFT-star-formation/arkenstone-method-spec.md; set-up
and every departure from S24a in arkenstone-test-va.md there. A static NFW halo with
the Diemer & Kravtsov (2014) outer term (--with-ext-potential=nfw-dk; M200 = 1e11 Msun,
c = 10, Planck 2018), live stellar disc and bulge, a 2e9 Msun gas disc and the rotating
cooling-flow CGM of Stern et al. (2023) to 600 kpc at 0.1 Z_sun, in a non-periodic
2.4 Mpc box. ICs from swift-ism-model tests/arkenstone/make_ics.py, read from
<data>/ics/ARK_<res>.hdf5.

Runs (SWIFT_REG_ARK_RUNS, comma-separated; default all), wind loadings of S24a Table 2
with eta_E = eta_E,kin + eta_E,th and kinetic_fraction = eta_E,kin / eta_E:
  nowind    Wind:loading_model none
  lowspec   eta_M = 6.41, eta_E = 0.966, kinetic fraction 0.8996 (Mach 4)
  highspec  eta_M = 0.32, eta_E = 0.900, kinetic fraction 0.3567 (Mach 1)
Resolution SWIFT_REG_ARK_RES = m2e5 (default; softening 0.2 kpc) or m1.6e6 (0.4 kpc).
Length SWIFT_REG_ARK_TEND_MYR (default 1000), snapshots every 10 Myr.

SPH:h_max = 120 kpc: halo particles need smoothing lengths of tens of kpc, and SWIFT
requires kernel_gamma * h_max * 1.1 below the top-level cell (2400 / 8 = 300 kpc).

Fluxes through shells at 0.1, 0.2, 0.5 and 1 r200 (r200 = 97.94 kpc) are thin-shell
particle sums, Mdot = sum m v_r / dr with dr = 0.1 r, energy with S24a eq. 10
(v^2/2 + c_s^2/(gamma-1)); SFR is the mass of stars formed in the preceding 20 Myr over
20 Myr (S24a Sec. 4.1.2).

Checks (each run reached its end time; then, where the runs exist):
  1. nowind: SFR averaged over 0.3 Gyr to the end equals the inflow through 0.1 r200
     averaged over the same snapshots within a factor 2. S24a Fig. 4 shows the two
     equal to tens of per cent; our ISM model and resolution differ from theirs, and
     the inflow measured on 10 Myr snapshots of a few hundred particles per shell
     scatters by tens of per cent, so only the factor is tested.
  2. highspec: eta_E at 0.1, 0.2 and 0.5 r200 over 0.2-1 Gyr, as time-averaged outflow
     energy rate over time-averaged u_* SFR, lies within a factor 3 of the input 0.9.
     S24a Fig. 6 shows 0.5-0.9 at all radii, with the enthalpy flux of eq. 10 (a Mach-1
     launch carries 1.43 times the input energy in that definition); a factor 3 allows
     for both. The time average is tested, not the median of the snapshot series: with
     whole-particle launches the energy arrives in quanta, and the median of 10 Myr
     samples measures how often a quantum is in the shell (at 1.6e6 Msun: median 0.005,
     time average 1.0 at 0.1 r200). Medians are recorded.
  3. highspec: time-averaged eta_M at 0.1 r200 over 0.2-1 Gyr above the input 0.32
     (S24a: about five times the input, from swept-up CGM).
  4. highspec: mean inflow through 0.2 r200 over 0.5 Gyr to the end below the nowind
     run's (S24a: inflow suppressed at all radii).
  5. lowspec: mean SFR over 0.2-0.7 Gyr below the nowind run's (S24a Fig. 4).
Recorded: time series of SFR and fluxes per run in the results JSON.
"""

import os
import shutil

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "Arkenstone I isolated galaxy (S24a) with PRFM SF and the decoupled wind"
TIER = "long"
BUILDS = {"ark": [params.PRFM_COOLING_FLAG, "--with-chemistry=EAGLE",
                  "--with-tracers=EAGLE", "--with-stars=EAGLE",
                  "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                  "--with-ext-potential=nfw-dk", "--with-wind=TIGRESS"]}

RES = os.environ.get("SWIFT_REG_ARK_RES", "m2e5")
SOFTENING = {"m2e5": 0.2, "m1.6e6": 0.4, "m1.3e7": 0.8}[RES]
IC_NAME = f"ARK_{RES}.hdf5"
MYR_INTERNAL = units.MYR / units.UNIT_TIME
T_END_MYR = float(os.environ.get("SWIFT_REG_ARK_TEND_MYR", "1000"))
DT_SNAP_MYR = float(os.environ.get("SWIFT_REG_ARK_DTSNAP_MYR", "10"))
R200_KPC = 97.94
BOX_KPC = 2400.0
SHELLS = (0.1, 0.2, 0.5, 1.0)
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g

WINDS = {
    "nowind": {"loading_model": "none"},
    "lowspec": {"loading_model": "constant", "eta_M": 6.41, "eta_E": 0.966,
                "kinetic_fraction": 0.869 / 0.966},
    "highspec": {"loading_model": "constant", "eta_M": 0.32, "eta_E": 0.900,
                 "kinetic_fraction": 0.321 / 0.900},
}
_only = os.environ.get("SWIFT_REG_ARK_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def ic_file(data_dir):
    path = os.path.join(data_dir, "ics", IC_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}: make it with swift-ism-model tests/arkenstone/make_ics.py")
    return path


def parameters(data_dir, t_end_myr, dt_snap_myr, wind):
    """Parameter dictionary of one run; also used by arkenstone_va_quick."""
    soft = SOFTENING
    p = params.eagle_base(IC_NAME, t_end_myr * MYR_INTERNAL, data_dir,
                          dt_snap_myr * MYR_INTERNAL, dt_max=1e-2,
                          cooling=params.PRFM_COOLING)
    for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
        p.pop(sec)
    p["InitialConditions"].update({"periodic": 0, "stars_smoothing_length": 0.5})
    p["Snapshots"]["compression"] = 4
    p["Gravity"].update({"max_physical_DM_softening": soft,
                         "max_physical_baryon_softening": soft})
    p["SPH"].update({"h_min_ratio": 0.1, "h_max": 120.0, "minimal_temperature": 100.0})
    p["Scheduler"] = {"max_top_level_cells": 8, "tasks_per_cell": 200}
    # Element abundances are per particle in the ICs (disc solar, CGM 0.1 solar)
    p["EAGLEChemistry"]["init_abundance_metal"] = -1.0
    p["NFWDKPotential"] = {
        "useabspos": 0, "position": [0.0, 0.0, 0.0], "M_200": 10.0,
        "concentration": 10.0, "h": 0.6766, "Omega_m": 0.3111, "s_e": 1.5, "b_e": 1.0,
        "timestep_mult": 0.01, "epsilon": soft}
    p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
    p["PRFMStarFormation"] = dict(
        calibration="OK22", threshold="measured", threshold_n_H_p_cm3=0.13,
        use_dark_matter=1, disc_centre=[0.0, 0.0, 0.0], disc_centre_is_absolute=0,
        pixel_kpc=0.1, cylinder_radius_kpc=0.5, half_height_kpc=1.0,
        map_half_width_kpc=25.0)
    p["Wind"] = dict(wind, launch_direction=[0.0, 0.0, 1.0],
                     recoupling_distance_kpc=2.0, recoupling_kernel_factor=2.0,
                     recoupling_time_Myr=25.0, max_timestep_Myr=1.0)
    return p


FLAGS = ["--hydro", "--self-gravity", "--external-gravity", "--stars",
         "--star-formation", "--cooling", "--limiter", "--sync"]


def run(ctx):
    for t in (params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    ics = ic_file(ctx.data_dir)
    for name in RUNS:
        os.makedirs(ctx.path(name))
        shutil.copy(ics, ctx.path(name, IC_NAME))
        p = parameters(ctx.data_dir, T_END_MYR, DT_SNAP_MYR, WINDS[name])
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=48 * 3600)


def shell_fluxes(snap, centre):
    """{r/r200: (Mdot_out, Mdot_in, Edot_out, Edot_kin_out, Edot_th_out)} in g/s and
    erg/s, thin-shell sums over gas particles (decoupled wind particles excluded)."""
    x = snap.read(0, "Coordinates") - centre
    v = snap.read(0, "Velocities")
    m = snap.read(0, "Masses")
    u = snap.read(0, "InternalEnergies")
    if snap.has(0, "WindDecoupled"):
        keep = snap.read(0, "WindDecoupled", cgs=False) == 0
        x, v, m, u = x[keep], v[keep], m[keep], u[keep]
    r = np.linalg.norm(x, axis=1)
    vr = np.sum(x * v, axis=1) / np.maximum(r, 1e-30)
    v2 = np.sum(v * v, axis=1)
    cs2_over = units.GAMMA * u  # c_s^2 / (gamma - 1) = gamma u
    out = {}
    for f in SHELLS:
        r0 = f * R200_KPC * units.KPC
        dr = 0.1 * r0
        s = np.abs(r - r0) < 0.5 * dr
        o, i = s & (vr > 0), s & (vr < 0)
        ek = m * vr * 0.5 * v2 / dr
        et = m * vr * cs2_over / dr
        out[f] = (np.sum(m[o] * vr[o]) / dr, -np.sum(m[i] * vr[i]) / dr,
                  np.sum(ek[o] + et[o]), np.sum(ek[o]), np.sum(et[o]))
    return out


def series(run_dir):
    """Times [Myr], SFR [Msun/yr] over the preceding 20 Myr, and shell fluxes."""
    snaps = snapshots(run_dir)
    last = snaps[-1]
    with_t = last.read(4, "BirthTimes") / units.MYR
    m0 = last.read(4, "InitialMasses") / units.SOLAR_MASS
    new = with_t > 0
    centre = np.full(3, 0.5 * BOX_KPC * units.KPC)
    t, sfr, fl = [], [], []
    for s in snaps[1:]:
        tm = s.time_cgs / units.MYR
        w = new & (with_t > tm - 20.0) & (with_t <= tm)
        t.append(tm)
        sfr.append(m0[w].sum() / 20e6)
        fl.append(shell_fluxes(s, centre))
    to_msun_yr = units.YEAR / units.SOLAR_MASS
    res = {"t": np.array(t), "sfr": np.array(sfr)}
    for f in SHELLS:
        a = np.array([x[f] for x in fl])
        res[f"mout_{f}"] = a[:, 0] * to_msun_yr
        res[f"min_{f}"] = a[:, 1] * to_msun_yr
        res[f"eout_{f}"] = a[:, 2] * units.YEAR
        res[f"ekin_{f}"] = a[:, 3] * units.YEAR
        res[f"eth_{f}"] = a[:, 4] * units.YEAR
    return res, last


def analyze(ctx):
    checks, data = [], {}
    for name in RUNS:
        res, last = series(ctx.path(name))
        data[name] = res
        checks.append(Check(f"{name}: final time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-2, source="run completed"))
        ctx.info[name] = {k: np.asarray(v).tolist() for k, v in res.items()}

    def window(res, t0, t1=None):
        t = res["t"]
        return (t >= t0) & (t <= (t1 if t1 is not None else t[-1] + 1))

    if "nowind" in data:
        d = data["nowind"]
        w = window(d, 300.0)
        if w.any():
            ratio = d["sfr"][w].mean() / max(d["min_0.1"][w].mean(), 1e-30)
            checks.append(Check("nowind: <SFR> / <inflow at 0.1 r200> >= 0.5", ratio, 0.5,
                                kind="min", source="S24a Fig. 4: SFR = inflow; factor 2"))
            checks.append(Check("nowind: <SFR> / <inflow at 0.1 r200> <= 2", ratio, 2.0,
                                kind="max", source="as above"))
    if "highspec" in data:
        d = data["highspec"]
        w = window(d, 200.0, 1000.0)
        ok = w & (d["sfr"] > 0)
        sfr = d["sfr"][w].mean() if w.any() else np.nan
        medians = {}
        for f in SHELLS[:3]:
            eta_E = d[f"eout_{f}"][w].mean() / (U_STAR * sfr * units.SOLAR_MASS)
            medians[f"eta_E_{f}"] = float(np.median(
                d[f"eout_{f}"][ok] / (U_STAR * d["sfr"][ok] * units.SOLAR_MASS)))
            checks.append(Check(f"highspec: eta_E at {f} r200 (0.2-1 Gyr) >= 0.3", eta_E,
                                0.3, kind="min", source="input 0.9 / 3 (S24a Fig. 6)"))
            checks.append(Check(f"highspec: eta_E at {f} r200 (0.2-1 Gyr) <= 2.7", eta_E,
                                2.7, kind="max", source="input 0.9 x 3"))
        checks.append(Check("highspec: eta_M at 0.1 r200 (0.2-1 Gyr) > input 0.32",
                            d["mout_0.1"][w].mean() / sfr, 0.32, kind="min",
                            source="S24a Fig. 6: ~5x input"))
        ctx.info["highspec_snapshot_medians_0.2-1Gyr"] = medians
        if "nowind" in data:
            n = data["nowind"]
            wh, wn = window(d, 500.0), window(n, 500.0)
            if wh.any() and wn.any():
                checks.append(Check("highspec: <inflow at 0.2 r200> below nowind",
                                    d["min_0.2"][wh].mean(), n["min_0.2"][wn].mean(),
                                    kind="max", source="S24a Fig. 4: inflow suppressed"))
    if "lowspec" in data and "nowind" in data:
        lo, n = data["lowspec"], data["nowind"]
        wl, wn = window(lo, 200.0, 700.0), window(n, 200.0, 700.0)
        if wl.any() and wn.any():
            checks.append(Check("lowspec: <SFR> over 0.2-0.7 Gyr below nowind",
                                lo["sfr"][wl].mean(), n["sfr"][wn].mean(), kind="max",
                                source="S24a Fig. 4: ejective suppression"))
    return checks
