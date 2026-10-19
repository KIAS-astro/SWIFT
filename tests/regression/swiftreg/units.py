"""Physical constants (cgs) and the internal unit system used by the tests.

The constants match src/physical_constants_cgs.h, so that expected values
computed here agree with the code to its own precision.
"""

import math

# Physical constants in cgs, from src/physical_constants_cgs.h
G_NEWTON = 6.67430e-8
PROTON_MASS = 1.67262192369e-24
BOLTZMANN_K = 1.380649e-16
YEAR = 3.15569251e7
PARSEC = 3.08567758149e18
SOLAR_MASS = 1.98841e33

KPC = 1e3 * PARSEC
MYR = 1e6 * YEAR
GYR = 1e9 * YEAR

# Internal units of every test: 1e10 Msun, 1 kpc, 1 km/s, 1 K
UNIT_MASS = 1.98841e43
UNIT_LENGTH = 3.08567758e21
UNIT_VELOCITY = 1e5
UNIT_CURRENT = 1.0
UNIT_TEMPERATURE = 1.0
UNIT_TIME = UNIT_LENGTH / UNIT_VELOCITY
UNIT_DENSITY = UNIT_MASS / UNIT_LENGTH ** 3
UNIT_ENERGY_PER_MASS = UNIT_VELOCITY ** 2

# Primordial hydrogen mass fraction and adiabatic index used by EAGLE runs
HYDROGEN_FRACTION = 0.752
GAMMA = 5.0 / 3.0

# Mean molecular weights used by SWIFT's hydro_props (neutral and ionised)
MU_NEUTRAL = 4.0 / (1.0 + 3.0 * HYDROGEN_FRACTION)
MU_IONISED = 4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_FRACTION))


def internal_unit_params():
    """Return the InternalUnitSystem section for a parameter file."""
    return {
        "UnitMass_in_cgs": UNIT_MASS,
        "UnitLength_in_cgs": UNIT_LENGTH,
        "UnitVelocity_in_cgs": UNIT_VELOCITY,
        "UnitCurrent_in_cgs": UNIT_CURRENT,
        "UnitTemp_in_cgs": UNIT_TEMPERATURE,
    }


def rho_from_nH(nH_cgs, X=HYDROGEN_FRACTION):
    """Mass density in g/cm^3 for a hydrogen number density in cm^-3."""
    return nH_cgs * PROTON_MASS / X


def u_from_T(T, mu):
    """Specific internal energy in erg/g for temperature T and weight mu."""
    return BOLTZMANN_K * T / ((GAMMA - 1.0) * mu * PROTON_MASS)


def T_from_u(u_cgs, mu):
    """Temperature in K for specific internal energy u in erg/g."""
    return u_cgs * (GAMMA - 1.0) * mu * PROTON_MASS / BOLTZMANN_K


def sound_speed(u_cgs):
    """Adiabatic sound speed in cm/s for specific internal energy in erg/g."""
    return math.sqrt(GAMMA * (GAMMA - 1.0) * u_cgs)
