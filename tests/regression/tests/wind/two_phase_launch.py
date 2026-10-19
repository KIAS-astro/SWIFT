"""Two-phase wind launch from PRFM star formation in a uniform gas layer.

The star-forming layer of wind/launch_rate (prfm_slab, Sigma_g = 200 Msun/pc^2)
with the warm channel on (wind-design.md Sec. 10). One uniform number per step
decides star, hot wind, warm wind or nothing (the four-way draw). Recoupling is
pushed beyond the run (d_rec = 1e3 kpc, t_rec = 1e4 Myr, and the warm apex is not
reached in 2 per cent of t_dep without gravity), so every launched particle is
still decoupled at the end and still carries its CR energy. Two runs:

  const  constant loadings: hot eta_M = 0.5, eta_E = 0.1; warm eta_M = 1.5;
         CR eta_E = 0.3 carried by the hot phase;
  tigpp  loading_model TIGRESS with loading_set TIGRESSpp: hot K20b x (0.515,
         0.40), warm 10^-0.628 Sigma_SFR^-0.44, CR eta_E = 0.3 carried by both
         phases, at each particle's PRFM Sigma_SFR.

Validation checks against the input loadings (each run):
  1. hot and warm launches against eta_M N_* (const) within 3 sigma; the
     numbers of a multinomial draw: N_x - eta_x N_* has variance
     eta_x (1 + eta_x) N_*;
  2. sum over launches of m / eta (the log's 'sum m/eta', with eta the loading
     of the launched phase at the launch Sigma_SFR) against the stellar mass
     formed, for each phase, within 3 sigma, sigma^2 = (1 + 1/eta) N_* m_g^2:
     an unbiased estimate whatever the spread of Sigma_SFR, so it tests the
     loading fits in tigpp;
  3. tigpp: the loadings recorded at launch equal the K26-anchored fits at the
     recorded Sigma_SFR (1e-5, float storage);
  4. warm launches: speed sqrt(2 e_kin) at launch has mean 100 km/s (1e-3; the
     gas velocity before launch averages out over the random signs) and
     internal energy u(1e4 K, mu = 0.6) (1e-6);
  5. CR energy per launched mass equals eta_E,CR u_* / eta_M,carry for every
     carrier (1e-5; eta_M,carry from the recorded float loadings), warm
     launches carry none in const;
  6. CR energy launched (log, double) equals the sum over the snapshot's
     decoupled particles (1e-12), and nothing was released;
  7. total CR energy launched against eta_E,CR u_* M_* within 3 sigma.
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _twophase as tp

DESCRIPTION = "two-phase wind: hot/warm launches per stellar mass = input loadings; CR energy carried exactly"
TIER = "quick"
BUILDS = {"wind": [params.PRFM_COOLING_FLAG, "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=constant", "--with-wind=TIGRESS"]}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

SIGMA = 200.0  # Msun / pc^2
FRACTION = 0.02
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g
V_WARM = 100.0e5  # cm/s
U_WARM = units.u_from_T(1e4, 0.6)
CR_ETA_E = 0.3

RUNS = {
    "const": {"loading_model": "constant", "eta_M": 0.5, "eta_E": 0.1,
              "warm_phase": 1, "warm_eta_M": 1.5, "cr_eta_E": CR_ETA_E,
              "cr_carriers": "hot"},
    "tigpp": {"loading_model": "TIGRESS", "loading_set": "TIGRESSpp",
              "warm_phase": 1, "cr_carriers": "both"},
}


def run(ctx):
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    for name, w in RUNS.items():
        os.makedirs(ctx.path(name), exist_ok=True)
        Sigma = slab.write_layer(ctx.path(name, "ics.hdf5"), SIGMA)
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
        p["Wind"] = dict(w, launch_direction=[0.0, 0.0, 1.0],
                         recoupling_distance_kpc=1e3, recoupling_time_Myr=1e4,
                         warm_recoupling_distance_kpc=1e3)
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=1800)


def analyze(ctx):
    checks = []
    for name, w in RUNS.items():
        last = snapshots(ctx.path(name))[-1]
        n_star = int(last.npart[4])
        m_star = last.read(4, "InitialMasses").sum()
        m_g = np.median(last.read(0, "Masses"))
        counts = last.read(0, "WindLaunchCounts", cgs=False)
        launched = counts > 0
        phase = last.read(0, "WindPhases", cgs=False)
        hot, warm = launched & (phase == 1), launched & (phase == 2)
        eta = last.read(0, "WindLaunchMassLoadings", cgs=False)
        sig = last.read(0, "WindLaunchSigmaSFR", cgs=False)
        tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
        to_g = units.UNIT_MASS

        if name == "const":
            for ph, sel, e in (("hot", hot, w["eta_M"]), ("warm", warm, w["warm_eta_M"])):
                checks.append(Check(f"{name}: {ph} launches / eta_M stars", int(sel.sum()),
                                    e * n_star, kind="sigma", tol=3.0,
                                    sigma=np.sqrt(e * (1 + e) * max(n_star, 1)),
                                    source=f"four-way draw, p_{ph} = {e} p_*; N_* = {n_star}"))
        for ph, sel in (("hot", hot), ("warm", warm)):
            e_med = float(np.median(eta[sel])) if sel.any() else 1.0
            est = tail[f"m_over_eta_{ph}"] * to_g
            checks.append(Check(f"{name}: sum m/eta ({ph}) / M_*", est / m_star, 1.0,
                                kind="sigma", tol=3.0,
                                sigma=np.sqrt((1 + 1 / e_med) * max(n_star, 1)) * m_g / m_star,
                                source="each launch stands for m/eta of stars in "
                                       "expectation (Horvitz-Thompson)"))
        if name == "tigpp":
            s = sig[hot]
            exp_h = np.array([tp.powerlaw(tp.HOT_M_NORM["TIGRESSpp"], tp.HOT_M_SLOPE, x)
                              for x in s])
            s2 = sig[warm]
            exp_w = np.array([tp.powerlaw(tp.WARM_NORM["TIGRESSpp"], tp.WARM_SLOPE, x)
                              for x in s2])
            checks.append(Check(f"{name}: max |eta_hot / fit - 1|",
                                np.abs(eta[hot] / exp_h - 1).max(), 1e-5, kind="max",
                                source="K20b hot x 0.515 at the launch Sigma_SFR"))
            checks.append(Check(f"{name}: max |eta_warm / fit - 1|",
                                np.abs(eta[warm] / exp_w - 1).max(), 1e-5, kind="max",
                                source="10^-0.628 Sigma_SFR^-0.44 (K26 crmhd 2 kpc)"))
            ctx.info[f"{name}_Sigma_SFR_launch_p16_50_84"] = np.percentile(
                np.concatenate([s, s2]), [16, 50, 84]).tolist()

        ek = last.read(0, "WindLaunchKineticEnergies")
        uw = last.read(0, "WindLaunchInternalEnergies")
        checks.append(Check(f"{name}: warm mean launch speed [km/s]",
                            np.sqrt(2 * ek[warm]).mean() / 1e5, V_WARM / 1e5, kind="rel",
                            tol=1e-3, source="Wind:warm_launch_speed_kms = 100"))
        checks.append(Check(f"{name}: warm max |u / u(1e4 K) - 1|",
                            np.abs(uw[warm] / U_WARM - 1).max(), 1e-6, kind="max",
                            source="T_warm = 1e4 K, mu = 0.6"))

        m = last.read(0, "Masses")
        E_cr = last.read(0, "WindCREnergies")
        if name == "const":
            carry = hot
            expect = CR_ETA_E * U_STAR / w["eta_M"] * np.ones(int(hot.sum()))
            checks.append(Check(f"{name}: CR energy on warm launches", float(E_cr[warm].sum()),
                                0.0, kind="abs", tol=0.0, source="cr_carriers: hot"))
        else:
            carry = hot | warm
            # eta_M,carry = eta_hot + eta_warm at the launch Sigma_SFR
            s_all = sig[carry]
            ec = np.array([tp.powerlaw(tp.HOT_M_NORM["TIGRESSpp"], tp.HOT_M_SLOPE, x)
                           + tp.powerlaw(tp.WARM_NORM["TIGRESSpp"], tp.WARM_SLOPE, x)
                           for x in s_all])
            expect = CR_ETA_E * U_STAR / ec
        checks.append(Check(f"{name}: max |E_CR / (m e_CR) - 1|",
                            np.abs(E_cr[carry] / (m[carry] * expect) - 1).max(), 1e-5,
                            kind="max", source="e_CR = eta_E,CR u_* / eta_M,carry"))
        E_log = tail["E_cr_launch"] * units.UNIT_MASS * units.UNIT_VELOCITY ** 2
        checks.append(Check(f"{name}: CR launched (log) / snapshot sum", E_log / E_cr.sum(),
                            1.0, kind="rel", tol=1e-12, source="double sums; none released"))
        checks.append(Check(f"{name}: CR released", tail["E_cr_release"], 0.0, kind="abs",
                            tol=0.0, source="no recoupling in the run"))
        eta_carry_med = float(np.median(m[carry] * U_STAR * CR_ETA_E / E_cr[carry]))
        checks.append(Check(f"{name}: CR launched / eta_E,CR u_* M_*",
                            E_cr.sum() / (CR_ETA_E * U_STAR * m_star), 1.0, kind="sigma",
                            tol=3.0,
                            sigma=np.sqrt((1 + 1 / eta_carry_med) * max(n_star, 1)) * m_g
                            / m_star,
                            source="Edot_CR = eta_E,CR u_* SFR in expectation"))
        ctx.info[name] = {"stars": n_star, "hot": int(hot.sum()), "warm": int(warm.sum()),
                          "log": tail}
    return checks
