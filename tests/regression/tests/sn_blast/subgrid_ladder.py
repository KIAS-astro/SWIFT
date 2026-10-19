"""One SN through each SN feedback scheme at gas particle masses 1-1000 Msun (long).

The small test problem of ai-notes sn-experiments.md Sec. 3. Uniform gas, n_H = 1
cm^-3, T = 1e4 K, solar metallicity, EAGLE cooling with the entropy floor off;
32^3 particles in a box set by the particle mass (98, 212, 456, 983 pc for
m_g = 1, 10, 100, 1000 Msun). One star particle at the centre, of initial mass
1 / N_SNII(8-100 Msun, Chabrier) = 85 Msun, releases 1e51 erg at once at 0.1 Myr
(fixed delay, f_E = 1, enrichment off) through the EAGLE-thermal-kinetic module:

  dump   f_kin = 0, Delta T = 1e3 K: p_heat > 1, every kernel neighbour shares E
  stoch  f_kin = 0, Delta T = 10^7.5 K: EAGLE stochastic heating
  kin    f_kin = 1, Delta v = 50 km/s: paired kicks (Chaikin+23 with f_kin = 1)
  tk     f_kin = 0.1, Delta T = 10^7.5 K, Delta v = 50 km/s: COLIBRE-like, fixed dT

stoch and tk at m_g = 100 and 1000 Msun expect 0.77 and 0.077 heating events per
SN, so they run 16 realisations (star IDs differ, which reseeds the draws).

Recorded for each run: radial momentum about the star p(t), mass with T > 5e4 K,
total gas energy, heating events. No pass criterion yet (sn-experiments.md): the
checks only confirm that each run reached 450 kyr. The momentum at 419 kyr
(10 t_sf of KO15) is compared, in the figure, with the resolved SWIFT run
(sn_blast/single_sn_cooling, 3.86e5 Msun km/s) and with KO15 Eq. 29 (2.95e5).
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "one SN through dump, stochastic, kinetic and thermal-kinetic schemes, m_g = 1-1000 Msun"
TIER = "long"
BUILDS = {"tk": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
                 "--with-entropy-floor=EAGLE", "--with-stars=EAGLE",
                 "--with-star-formation=EAGLE", "--with-black-holes=EAGLE",
                 "--with-ext-potential=constant", "--with-number-of-SNII-rays=8",
                 "--with-feedback=EAGLE-thermal-kinetic"]}

N = 32
NH = 1.0
T = 1e4
M_GAS = (1.0, 10.0, 100.0, 1000.0)  # Msun
SCHEMES = {
    "dump": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 1e3},
    "stoch": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 3.16228e7},
    "kin": {"SNII_f_kinetic": 1.0, "SNII_delta_v_km_p_s": 50.0},
    "tk": {"SNII_f_kinetic": 0.1, "SNII_delta_T_K": 3.16228e7, "SNII_delta_v_km_p_s": 50.0},
}
STOCHASTIC = ("stoch", "tk")
N_REAL = 16
DT_SNAP = 1.0226e-5  # 10 kyr
T_END = 4.6e-4  # 450 kyr
T_HOT = 5e4


def box_kpc(m_gas):
    return (N ** 3 * m_gas * units.SOLAR_MASS / units.rho_from_nH(NH)) ** (1 / 3) / units.KPC


def realisations(scheme, m_gas):
    return N_REAL if scheme in STOCHASTIC and m_gas >= 100 else 1


def run_name(scheme, m_gas, k):
    return f"{scheme}_m{m_gas:g}_r{k}"


def run(ctx):
    from tests.feedback._common import snii_per_msun

    ctx.table("coolingtables")
    ctx.table("photometry")
    ctx.table("yieldtables")
    m_star = 1.0 / snii_per_msun()  # Msun, one SN
    ctx.info["m_star_Msun"] = m_star
    for m_gas in M_GAS:
        L = box_kpc(m_gas)
        for scheme, fb in SCHEMES.items():
            for k in range(realisations(scheme, m_gas)):
                name = run_name(scheme, m_gas, k)
                os.makedirs(ctx.path(name))
                gas, _ = ics.uniform_gas(N, L, NH, T)
                stars = {
                    "Coordinates": np.full((1, 3), 0.5 * L),
                    "Velocities": np.zeros((1, 3)),
                    "Masses": np.array([m_star * units.SOLAR_MASS / units.UNIT_MASS]),
                    "SmoothingLength": np.array([1.2348 * L / N]),
                    # different IDs reseed the stochastic draws
                    "ParticleIDs": np.array([N ** 3 + 1 + 7919 * k], dtype=np.uint64),
                }
                ics.write(ctx.path(name, "ics.hdf5"), L, gas=gas, stars=stars)
                p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-5)
                p["Snapshots"]["compression"] = 4
                p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
                p["Stars"].update({"overwrite_birth_time": 1, "birth_time": 0.0})
                # The hot cavity empties late in the run; without a cap a
                # smoothing length outgrows the top-level cells and SWIFT stops.
                p["SPH"]["h_max"] = L / 8
                p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                               "Cool_temperature_norm_K": 10.0})
                p["EAGLEFeedback"].update({
                    "use_SNIa_feedback": 0, "use_AGB_enrichment": 0,
                    "use_SNII_enrichment": 0, "use_SNIa_enrichment": 0,
                    "AGB_ejecta_velocity_km_p_s": 0.0,
                    "SNII_energy_fraction_min": 1.0, "SNII_energy_fraction_max": 1.0,
                    "SNII_sampled_delay": 0, "SNII_wind_delay_Gyr": 1e-4,
                })
                p["EAGLEFeedback"].update(fb)
                ctx.swift(p, ["--hydro", "--stars", "--feedback", "--external-gravity",
                              "--cooling", "--limiter", "--sync"],
                          subdir=name, name=name, timeout=3600)


def history(run_dir, L_kpc):
    """Rows of t [yr], p_r [Msun km/s], M(T > T_HOT) [Msun], E_gas [erg]."""
    box_cm = L_kpc * units.UNIT_LENGTH
    rows = []
    for s in snapshots(run_dir):
        d = c.radial(s, box_cm)
        p_r = np.sum(d["m"] * d["vr"]) / (units.SOLAR_MASS * 1e5)
        hot = np.sum(d["m"][d["T"] > T_HOT]) / units.SOLAR_MASS
        e = np.sum(d["m"] * d["u"]) + 0.5 * np.sum(d["m"] * np.sum(d["v"] ** 2, axis=1))
        rows.append([s.time_cgs / units.YEAR, p_r, hot, e])
    return np.array(rows)


def analyze(ctx):
    checks = []
    for m_gas in M_GAS:
        L = box_kpc(m_gas)
        for scheme in SCHEMES:
            for k in range(realisations(scheme, m_gas)):
                name = run_name(scheme, m_gas, k)
                rows = history(ctx.path(name), L)
                last = snapshots(ctx.path(name))[-1]
                n_heat = float(np.sum(last.read(4, "FeedbackNumberOfHeatingEvents", cgs=False)))
                # energy jump across the injection at 0.1 Myr, outputs at 90 and 110
                # kyr; a lower bound, since cooling acts within those 10 kyr
                e_inj = (np.interp(1.1e5, rows[:, 0], rows[:, 3])
                         - np.interp(0.9e5, rows[:, 0], rows[:, 3]))
                ctx.info[name] = {"t_yr_pr_Mhot_E": rows.tolist(), "n_heat": n_heat,
                                  "E_injected_estimate": e_inj}
                checks.append(Check(f"{name}: final time [kyr]", rows[-1, 0] / 1e3,
                                    T_END * units.UNIT_TIME / units.YEAR / 1e3,
                                    kind="rel", tol=1e-3, source="run completed"))
    return checks
