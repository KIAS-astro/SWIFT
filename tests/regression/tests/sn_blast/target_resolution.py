"""One star particle's feedback at the target resolution, m_g = 1.84e6 Msun (long).

The science runs will use baryon particles of a few times 1e6 Msun (COLIBRE m6:
1.84e6 Msun). There a star particle of the gas particle mass releases
m_* x 0.0118 = 2.2e4 CCSNe (Chabrier IMF, 8-100 Msun) over 3-40 Myr, and the
question is how much energy and momentum each scheme leaves in the gas.

Set-up: uniform periodic gas, n_H = 0.1, 1, 10, 100 cm^-3, solar metallicity, EAGLE
cooling with the EAGLE entropy floor (as in production), 32^3 particles of
1.84e6 Msun (boxes of 26, 12, 5.6 and 2.6 kpc). The DVS12 critical density for
Delta T = 10^7.5 K at this mass is about 6 cm^-3, so the set spans both sides; one star particle of the same mass at
the centre, born at t = 0, CCSN energy released continuously (sampled delays),
f_E = 1 for every scheme so they share one energy budget, enrichment and SNIa off;
50 Myr. Schemes (EAGLE-thermal-kinetic module):
  none     no feedback: baseline for the ambient gas's own cooling
  dump     f_kin = 0, Delta T = 1e3 K (p_heat > 1: every kernel neighbour shares E)
  kin      f_kin = 1, 50 km/s pair kicks
  stoch    f_kin = 0, Delta T = 10^7.5 K (EAGLE)
  tk       f_kin = 0.1, Delta T = 10^7.5 K, 50 km/s
  colibre  f_kin = 0.1, Delta T(n_H) with the m6 values of Schaye+26 Table 1
At this mass one heating event at 10^7.5 K carries 2.4e55 erg, about the star's
whole CCSN budget (2.2e55 erg), so a star expects about one event in its life
(Chaikin+23 eq. 4) and a large fraction of realisations get none; the stochastic
schemes therefore run N_REAL = 64 realisations (star IDs differ). Each run's
history is computed as soon as it ends and its snapshots are deleted, except for
realisation 0, which is kept for slices and movies (64 realisations x 50 outputs
would otherwise need ~40 GB).

Recorded per run: radial momentum about the star, mass with T > 1e5 K, total gas
energy, heating events, every 1 Myr. No pass criterion: this measures the schemes.
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "one m6 star particle (2.2e4 SNe) through each scheme at n_H = 0.1, 1, 10"
TIER = "long"
BUILDS = {"tk": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
                 "--with-entropy-floor=EAGLE", "--with-stars=EAGLE",
                 "--with-star-formation=EAGLE", "--with-black-holes=EAGLE",
                 "--with-ext-potential=constant", "--with-number-of-SNII-rays=8",
                 "--with-feedback=EAGLE-thermal-kinetic"]}

N = 32
M_GAS = 1.84e6  # Msun
DENSITIES = (0.1, 1.0, 10.0, 100.0)
T_INIT = 8e3  # K, on or above the EAGLE floor at these densities
COLIBRE_M6 = {"SNII_delta_T_model": "DensityDependent", "SNII_delta_T_K": 10 ** 6.5,
              "SNII_delta_T_n_pivot_H_p_cm3": 0.5, "SNII_delta_T_min_K": 10 ** 6.75,
              "SNII_delta_T_max_K": 1e8}
SCHEMES = {
    "none": None,
    "dump": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 1e3},
    "kin": {"SNII_f_kinetic": 1.0, "SNII_delta_v_km_p_s": 50.0},
    "stoch": {"SNII_f_kinetic": 0.0, "SNII_delta_T_K": 10 ** 7.5},
    "tk": {"SNII_f_kinetic": 0.1, "SNII_delta_T_K": 10 ** 7.5, "SNII_delta_v_km_p_s": 50.0},
    "colibre": dict(COLIBRE_M6, SNII_f_kinetic=0.1, SNII_delta_v_km_p_s=50.0),
}
STOCHASTIC = ("stoch", "tk", "colibre")
N_REAL = 64
DT_SNAP = 1.0226e-3  # 1 Myr
T_END = 5.113e-2  # 50 Myr
T_HOT = 1e5


def box_kpc(nH):
    return (N ** 3 * M_GAS * units.SOLAR_MASS / units.rho_from_nH(nH)) ** (1 / 3) / units.KPC


def run_name(scheme, nH, k):
    return f"{scheme}_n{nH:g}_r{k}"


def runs():
    for nH in DENSITIES:
        for scheme in SCHEMES:
            for k in range(N_REAL if scheme in STOCHASTIC else 1):
                yield scheme, nH, k


def run(ctx):
    from swiftreg import data
    ctx.table("coolingtables")
    ctx.table("photometry")
    ctx.table("yieldtables")
    glass = data.glass(32, ctx.data_dir, log=ctx.log)
    for scheme, nH, k in runs():
        L = box_kpc(nH)
        name = run_name(scheme, nH, k)
        os.makedirs(ctx.path(name))
        gas, _ = ics.uniform_gas(N, L, nH, T_INIT, glass_file=glass)
        stars = {
            "Coordinates": np.full((1, 3), 0.5 * L),
            "Velocities": np.zeros((1, 3)),
            "Masses": np.array([M_GAS * units.SOLAR_MASS / units.UNIT_MASS]),
            "SmoothingLength": np.array([1.2348 * L / N]),
            "ParticleIDs": np.array([N ** 3 + 1 + 7919 * k], dtype=np.uint64),
        }
        ics.write(ctx.path(name, "ics.hdf5"), L, gas=gas, stars=stars)
        p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-3)
        p["Snapshots"]["compression"] = 4
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
        p["Stars"].update({"overwrite_birth_time": 1, "birth_time": 0.0})
        p["SPH"]["h_max"] = L / 8
        p["EAGLEFeedback"].update({
            "use_SNIa_feedback": 0, "use_AGB_enrichment": 0, "use_SNII_enrichment": 0,
            "use_SNIa_enrichment": 0, "AGB_ejecta_velocity_km_p_s": 0.0,
            "SNII_energy_fraction_min": 1.0, "SNII_energy_fraction_max": 1.0,
            "SNII_sampled_delay": 1,
        })
        flags = ["--hydro", "--stars", "--external-gravity", "--cooling", "--limiter",
                 "--sync"]
        if SCHEMES[scheme] is not None:
            p["EAGLEFeedback"].update(SCHEMES[scheme])
            flags.append("--feedback")
        ctx.swift(p, flags, subdir=name, name=name, timeout=7200)
        rows = history(ctx.path(name), L)
        last = snapshots(ctx.path(name))[-1]
        n_heat = float(np.sum(last.read(4, "FeedbackNumberOfHeatingEvents", cgs=False)))
        ctx.info[name] = {"t_Myr_pr_Mhot_E": rows.tolist(), "n_heat": n_heat}
        if k > 0:
            for f in os.listdir(ctx.path(name)):
                if f.startswith("snap_") and f.endswith(".hdf5"):
                    os.remove(ctx.path(name, f))


def history(run_dir, L_kpc):
    """Rows of t [Myr], p_r [Msun km/s], M(T > T_HOT) [Msun], E_gas [erg]."""
    box_cm = L_kpc * units.UNIT_LENGTH
    rows = []
    for s in snapshots(run_dir):
        d = c.radial(s, box_cm)
        p_r = np.sum(d["m"] * d["vr"]) / (units.SOLAR_MASS * 1e5)
        hot = np.sum(d["m"][d["T"] > T_HOT]) / units.SOLAR_MASS
        e = np.sum(d["m"] * d["u"]) + 0.5 * np.sum(d["m"] * np.sum(d["v"] ** 2, axis=1))
        rows.append([s.time_cgs / units.MYR, p_r, hot, e])
    return np.array(rows)


def analyze(ctx):
    checks = []
    for scheme, nH, k in runs():
        name = run_name(scheme, nH, k)
        rows = np.array(ctx.info[name]["t_Myr_pr_Mhot_E"])
        checks.append(Check(f"{name}: final time [Myr]", rows[-1, 0],
                            T_END * units.UNIT_TIME / units.MYR, kind="rel", tol=1e-3,
                            source="run completed"))
    return checks
