"""Shared set-up for the EAGLE star formation tests.

The expected star formation rate is the pressure law of Schaye & Dalla
Vecchia (2008), their eq. 19, as implemented in
src/star_formation/EAGLE/star_formation.h:

    mdot = m_g A (1 Msun pc^-2)^-n (gamma f_g P / G)^((n-1)/2)

with A = KS_normalisation, n = KS_exponent, f_g = 1, gamma = 5/3, and P
the particle's own SPH pressure (not the entropy-floor pressure).
"""

import os

from swiftreg import ics, params, units

# Star particles require gravity; a zero constant external field satisfies
# that without letting the cold box collapse under self-gravity.
EAGLE_BUILD = ["--with-subgrid=EAGLE", "--with-ext-potential=constant"]
# --temperature (required by the EAGLE tracers) computes temperatures from
# the cooling tables without applying cooling, so u changes only by hydro.
SF_FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity",
            "--temperature"]

N = 32
BOX = 1.0  # kpc
Z = params.EAGLE_SOLAR_ABUNDANCES["init_abundance_metal"]


def sf_params(ctx, time_end, snapshot_dt):
    base = params.eagle_base(
        "ics.hdf5", time_end, ctx.data_dir, snapshot_dt, dt_max=1e-2
    )
    base["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
    # As gas converts, smoothing lengths grow; SWIFT cannot reduce the number
    # of top-level cells during a run, so start with few and cap h.
    base["Scheduler"] = {"max_top_level_cells": 8}
    base["SPH"]["h_max"] = 2.0 * 1.2348 * BOX / N
    # Stars need the photometry tables, and --temperature the cooling tables,
    # at start-up even without feedback or cooling.
    ctx.table("photometry")
    ctx.table("coolingtables")
    return base


def write_box(ctx, nH, T, subdir=None):
    gas, mgas_msun = ics.uniform_gas(N, BOX, nH, T)
    path = ctx.path(subdir, "ics.hdf5") if subdir else ctx.path("ics.hdf5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ics.write(path, BOX, gas=gas)
    return mgas_msun


def sfr_pressure_law(mass_cgs, pressure_cgs, sf):
    """Expected SFR in g/s from the EAGLEStarFormation parameter section."""
    A = sf["KS_normalisation"] * units.SOLAR_MASS / units.YEAR / units.KPC ** 2
    n = sf["KS_exponent"]
    sigma_unit = units.SOLAR_MASS / units.PARSEC ** 2
    fg = sf.get("gas_fraction", 1.0)
    return (
        mass_cgs
        * A
        * sigma_unit ** (-n)
        * (units.GAMMA * fg * pressure_cgs / units.G_NEWTON) ** ((n - 1.0) / 2.0)
    )


def density_threshold(sf, Z):
    """Metallicity-dependent threshold n_H in cm^-3 (Schaye 2004)."""
    nth = sf["threshold_norm_H_p_cm3"] * (Z / sf["threshold_Z0"]) ** sf["threshold_slope"]
    return min(nth, sf["threshold_max_density_H_p_cm3"])
