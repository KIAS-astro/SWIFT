"""Parameter files: nested dictionaries written in SWIFT's YAML dialect.

SWIFT's parser reads `Section:` lines followed by indented `key: value`
lines, with lists written inline as `[a, b, c]`. A general YAML emitter
writes lists as block sequences, which SWIFT does not read, so the files
are written here directly.
"""

import copy

from . import units


def _format_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write(params, path):
    """Write a {section: {key: value}} dictionary as a SWIFT parameter file."""
    with open(path, "w") as f:
        for section, entries in params.items():
            f.write(f"{section}:\n")
            for key, value in entries.items():
                f.write(f"  {key}: {_format_value(value)}\n")
            f.write("\n")


def merge(base, overrides):
    """Return a deep copy of base with overrides applied section by section.

    A value of None removes the key; a section mapped to None is removed.
    """
    out = copy.deepcopy(base)
    for section, entries in overrides.items():
        if entries is None:
            out.pop(section, None)
            continue
        target = out.setdefault(section, {})
        for key, value in entries.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    return out


# Solar abundances used by the EAGLE examples (mass fractions)
EAGLE_SOLAR_ABUNDANCES = {
    "init_abundance_metal": 0.0129,
    "init_abundance_Hydrogen": 0.7065,
    "init_abundance_Helium": 0.2806,
    "init_abundance_Carbon": 0.00207,
    "init_abundance_Nitrogen": 0.000836,
    "init_abundance_Oxygen": 0.00549,
    "init_abundance_Neon": 0.00141,
    "init_abundance_Magnesium": 0.000591,
    "init_abundance_Silicon": 0.000683,
    "init_abundance_Iron": 0.0011,
}


def eagle_base(ic_file, time_end, data_dir, snapshot_dt, dt_max=1e-3):
    """Parameters for a non-cosmological, periodic EAGLE-physics test box.

    The physics sections follow examples/IsolatedGalaxy/IsolatedGalaxy_feedback
    (EAGLE reference values). The entropy floor uses the EAGLE reference
    model of Schaye et al. (2015): Jeans floor 8000 K at n_H = 0.1 cm^-3
    with slope 4/3, and an 8000 K isothermal floor above n_H = 1e-5 cm^-3.
    Tests override what they need with merge().
    """
    return {
        "MetaData": {"run_name": "swift-regression"},
        "InternalUnitSystem": units.internal_unit_params(),
        "TimeIntegration": {
            "time_begin": 0.0,
            "time_end": float(time_end),
            "dt_min": 1e-12,
            "dt_max": float(dt_max),
        },
        "Snapshots": {
            "basename": "snap",
            "time_first": 0.0,
            "delta_time": float(snapshot_dt),
            "recording_triggers_part": [-1, -1],
            "recording_triggers_bpart": [-1, -1],
        },
        "Statistics": {"delta_time": float(snapshot_dt), "time_first": 0.0},
        # Few top-level cells: test boxes are small, smoothing lengths grow
        # when gas converts or is heated, and SWIFT cannot reduce the number
        # of top-level cells during a run.
        "Scheduler": {"max_top_level_cells": 8, "tasks_per_cell": 200},
        "InitialConditions": {
            "file_name": ic_file,
            "periodic": 1,
        },
        "SPH": {
            "resolution_eta": 1.2348,
            "CFL_condition": 0.1,
            "h_min_ratio": 0.0,
            "minimal_temperature": 10.0,
        },
        "Stars": {
            "resolution_eta": 1.1642,
            "h_tolerance": 7e-3,
            "timestep_age_threshold_Myr": 10.0,
            "max_timestep_young_Myr": 0.1,
            "max_timestep_old_Myr": 1.0,
            "luminosity_filename": f"{data_dir}/photometry",
        },
        "Gravity": {
            "eta": 0.025,
            "MAC": "geometric",
            "theta_cr": 0.7,
            "use_tree_below_softening": 1,
            "max_physical_DM_softening": 0.01,
            "max_physical_baryon_softening": 0.01,
        },
        "EAGLECooling": {
            "dir_name": f"{data_dir}/coolingtables/",
            "H_reion_z": 7.5,
            "H_reion_eV_p_H": 2.0,
            "He_reion_z_centre": 3.5,
            "He_reion_z_sigma": 0.5,
            "He_reion_eV_p_H": 2.0,
        },
        "EAGLEChemistry": dict(EAGLE_SOLAR_ABUNDANCES),
        "EAGLEStarFormation": {
            "SF_threshold": "Zdep",
            "SF_model": "PressureLaw",
            "KS_normalisation": 1.515e-4,
            "KS_exponent": 1.4,
            "min_over_density": 100.0,
            "KS_high_density_threshold_H_p_cm3": 1e3,
            "KS_high_density_exponent": 2.0,
            "EOS_entropy_margin_dex": 0.5,
            "threshold_norm_H_p_cm3": 0.1,
            "threshold_Z0": 0.002,
            "threshold_slope": -0.64,
            "threshold_max_density_H_p_cm3": 10.0,
        },
        "EAGLEEntropyFloor": {
            "Jeans_density_threshold_H_p_cm3": 0.1,
            "Jeans_over_density_threshold": 10.0,
            "Jeans_temperature_norm_K": 8000.0,
            "Jeans_gamma_effective": 1.3333333,
            "Cool_density_threshold_H_p_cm3": 1e-5,
            "Cool_over_density_threshold": 10.0,
            "Cool_temperature_norm_K": 8000.0,
            "Cool_gamma_effective": 1.0,
        },
        "EAGLEFeedback": {
            "use_SNII_feedback": 1,
            "use_SNIa_feedback": 1,
            "use_AGB_enrichment": 1,
            "use_SNII_enrichment": 1,
            "use_SNIa_enrichment": 1,
            "filename": f"{data_dir}/yieldtables/",
            "IMF_min_mass_Msun": 0.1,
            "IMF_max_mass_Msun": 100.0,
            "SNII_min_mass_Msun": 8.0,
            "SNII_max_mass_Msun": 100.0,
            "SNII_feedback_model": "Isotropic",
            "SNII_sampled_delay": 1,
            "SNII_delta_T_K": 3.16228e7,
            "SNII_delta_v_km_p_s": 200.0,
            "SNII_energy_erg": 1.0e51,
            "SNII_energy_fraction_function": "EAGLE",
            "SNII_energy_fraction_min": 1.0,
            "SNII_energy_fraction_max": 1.0,
            "SNII_energy_fraction_Z_0": 0.0012663729,
            "SNII_energy_fraction_n_0_H_p_cm3": 1.4588,
            "SNII_energy_fraction_n_Z": 0.8686,
            "SNII_energy_fraction_n_n": 0.8686,
            "SNII_energy_fraction_use_birth_density": 0,
            "SNII_energy_fraction_use_birth_metallicity": 0,
            "SNIa_DTD": "Exponential",
            "SNIa_DTD_delay_Gyr": 0.04,
            "SNIa_DTD_exp_timescale_Gyr": 2.0,
            "SNIa_DTD_exp_norm_p_Msun": 0.002,
            "SNIa_energy_erg": 1.0e51,
            "AGB_ejecta_velocity_km_p_s": 10.0,
            "stellar_evolution_age_cut_Gyr": 0.1,
            "stellar_evolution_sampling_rate": 10,
        },
    }
