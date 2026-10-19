"""Shared set-up for the SN feedback energy-budget tests.

A uniform periodic box of gas at rest (n_H = 1 cm^-3, T = 1e4 K, no cooling,
no self-gravity) holds N_STARS star particles at random positions, all born
at t = 0. Only CCSN energy feedback is on: SNIa energy, all enrichment (so
gas masses stay fixed) and AGB ejecta kinetic energy are switched off, and
f_E = 1. The run covers 50 Myr, longer than the lifetime of the least
massive CCSN progenitor (8 Msun, about 40 Myr at solar metallicity), so the
whole CCSN budget is released:

    E_SN = f_E * 1e51 erg * m_* * N_SNII,   N_SNII = int_8^100 phi(m) dm,

with phi the Chabrier (2003) IMF of src/feedback/EAGLE/imf.h, normalised to
unit mass over 0.1-100 Msun. The injected energy is drawn stochastically,
event by event, so it matches E_SN only within the Poisson scatter of the
number of events.
"""


import numpy as np
from scipy.integrate import quad

from swiftreg import ics, params, units

N = 32
BOX = 1.0  # kpc
NH = 1.0
T_GAS = 1e4
N_STARS = 128
STAR_MASS_IN_GAS_MASSES = 20.0
T_END = 0.0512  # internal units, 50.1 Myr
N_SNAP = 25
SEED = 2026

FLAGS = ["--hydro", "--stars", "--feedback", "--external-gravity",
         "--temperature", "--limiter", "--sync"]


def chabrier(m):
    """EAGLE's Chabrier (2003) IMF, dN/dm up to normalisation."""
    if m > 1.0:
        return 0.237912 * m ** -2.3
    x = np.log10(m) - np.log10(0.079)
    return 0.852464 * np.exp(-x * x / (2 * 0.69 ** 2)) / m


def snii_per_msun(m_min=8.0, m_max=100.0):
    norm = quad(lambda m: m * chabrier(m), 0.1, 100.0, points=[1.0])[0]
    return quad(chabrier, m_min, m_max, points=[1.0] if m_min < 1 else None)[0] / norm


def setup(ctx, feedback_overrides):
    """Write ICs and parameters; return (params, gas mass, star mass) in cgs."""
    gas, mgas_msun = ics.uniform_gas(N, BOX, NH, T_GAS)
    rng = np.random.default_rng(SEED)
    mstar = STAR_MASS_IN_GAS_MASSES * gas["Masses"][0]
    stars = {
        "Coordinates": rng.uniform(0, BOX, size=(N_STARS, 3)),
        "Velocities": np.zeros((N_STARS, 3)),
        "Masses": np.full(N_STARS, mstar),
        "SmoothingLength": np.full(N_STARS, 1.2348 * BOX / N),
        "ParticleIDs": np.arange(N ** 3 + 1, N ** 3 + 1 + N_STARS, dtype=np.uint64),
    }
    ics.write(ctx.path("ics.hdf5"), BOX, gas=gas, stars=stars)

    ctx.table("photometry")
    ctx.table("coolingtables")
    ctx.table("yieldtables")
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, T_END / N_SNAP, dt_max=1e-3)
    p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
    p["Stars"].update({"overwrite_birth_time": 1, "birth_time": 0.0})
    fb = p["EAGLEFeedback"]
    fb.update({
        "use_SNIa_feedback": 0,
        "use_AGB_enrichment": 0,
        "use_SNII_enrichment": 0,
        "use_SNIa_enrichment": 0,
        "AGB_ejecta_velocity_km_p_s": 0.0,
        "SNII_energy_fraction_min": 1.0,
        "SNII_energy_fraction_max": 1.0,
        "SNII_sampled_delay": 1,
        "SNII_wind_delay_Gyr": 0.03,
    })
    fb.update(feedback_overrides)
    ctx.info["feedback"] = dict(fb)
    return p, gas["Masses"][0] * units.UNIT_MASS, mstar * units.UNIT_MASS


def total_energy(snap):
    """Thermal plus kinetic energy of the gas in erg."""
    m = snap.read(0, "Masses")
    u = snap.read(0, "InternalEnergies")
    v = snap.read(0, "Velocities")
    return np.sum(m * u) + 0.5 * np.sum(m * np.sum(v * v, axis=1))


def gas_momentum(snap):
    m = snap.read(0, "Masses")
    v = snap.read(0, "Velocities")
    return np.sum(m[:, None] * v, axis=0)


def expected_sn_energy(fb, mstar_cgs):
    n_sn = snii_per_msun(fb["SNII_min_mass_Msun"], fb["SNII_max_mass_Msun"])
    return fb["SNII_energy_fraction_max"] * fb["SNII_energy_erg"] * N_STARS * (
        mstar_cgs / units.SOLAR_MASS) * n_sn
