/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2018 Folkert Nobels (nobels@strw.leidenuniv.nl)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/
#ifndef SWIFT_PRFM_STAR_FORMATION_H
#define SWIFT_PRFM_STAR_FORMATION_H

/* Local includes */
#include "adiabatic_index.h"
#include "chemistry.h"
#include "cooling.h"
#include "cosmology.h"
#include "engine.h"
#include "entropy_floor.h"
#include "equation_of_state.h"
#include "exp10.h"
#include "hydro.h"
#include "parser.h"
#include "part.h"
#include "physical_constants.h"
#include "prfm_map.h"
#include "prfm_physics.h"
#include "random.h"
#include "stars.h"
#include "units.h"

#define star_formation_need_update_dx_max 0

/**
 * @file src/star_formation/PRFM/star_formation.h
 * @brief Pressure-regulated, feedback-modulated star formation (Ostriker &
 * Kim 2022), in the PRFM-int form of Jeffreson et al. (2026, Sec. 2.6).
 *
 * A gas particle with n_H = rho / (1.4 m_H) >= n_th (or, as an option, an
 * equilibrium density Sigma_g / (2 H_g 1.4 m_H) >= n_th together with a
 * lower measured n_H >= n_floor, J26 p. 23) and with T below
 * T_max or entropy within a margin of the entropy floor forms stars at
 * mdot = m_g / t_dep, t_dep = Upsilon_tot Sigma_g / W (src/prfm_physics.h).
 * Sigma_g, Sigma_* and sigma_*z come from the face-on map of the isolated
 * disc (src/prfm_map.h), rho_d from the Hernquist potential. Conversion is
 * stochastic as in the EAGLE module: probability SFR dt / m_g per step.
 */

/**
 * @brief Choice of density threshold.
 */
enum prfm_star_formation_threshold {
  prfm_threshold_measured,   /*<! n_H from the particle's density */
  prfm_threshold_equilibrium /*<! n_H,eq = Sigma_g / (2 H_g 1.4 m_H), and
                                  the measured n_H above a lower floor */
};

/**
 * @brief Properties of the PRFM star formation model.
 */
struct star_formation {

  /*! Calibration of yield and dispersion */
  struct prfm_calibration cal;

  /*! Which calibration (for printing) */
  enum prfm_calibration_choice cal_choice;

  /*! Dark matter coefficient c_d of J26 eq. 31 */
  double c_d;

  /*! Number of fixed-point iterations on sigma_eff */
  int n_iter;

  /*! Inverse of the solar metal mass fraction, for Z' */
  double solar_metallicity_inv;

  /*! Which density threshold */
  enum prfm_star_formation_threshold threshold;

  /*! Threshold n_th in H atoms per cm^3 */
  double n_th_H_p_cm3;

  /*! With the equilibrium threshold, the lowest measured n_H [cm^-3] */
  double n_floor_H_p_cm3;

  /*! Physical mass density to n_H in cm^-3 (n_H = rho / 1.4 m_H) */
  double rho_to_n_H_cgs;

  /*! Particles below this temperature [K] are eligible */
  double T_max_K;

  /*! Particles within this margin of the entropy floor are eligible [dex] */
  double entropy_margin_dex;

  /*! 10^entropy_margin_dex */
  double ten_to_entropy_margin_dex;

  /*! Internal to cgs: surface density, density, velocity */
  double Sigma_to_cgs, rho_to_cgs, velocity_to_cgs;

  /*! cgs time to internal units */
  double time_from_cgs;

  /*! cgs pressure and length to internal units */
  double pressure_from_cgs, length_from_cgs;

  /*! The face-on map */
  struct prfm_map_params map;

  /*! Number of stars to form per gas particle */
  int num_stars_per_gas_particle;
};

/**
 * @brief Metallicity of a particle in solar units.
 */
INLINE static double star_formation_prfm_Zprime(
    const struct part *p, const struct star_formation *starform) {
  return chemistry_get_total_metal_mass_fraction_for_star_formation(p) *
         starform->solar_metallicity_inv;
}

/**
 * @brief Evaluate PRFM-int at a particle from the face-on map.
 *
 * @param p The #part.
 * @param starform The star formation properties.
 * @param state (return) The PRFM state (cgs); unset if Sigma_g = 0.
 * @param xp (return) If not NULL, the map values and the state are stored
 * in its sf_data (internal units), or zeroed if Sigma_g = 0.
 * @return 0 if the particle sees no gas on the map, 1 otherwise.
 */
INLINE static int star_formation_prfm_state(
    const struct part *p, const struct star_formation *starform,
    struct prfm_state *state, struct xpart *xp) {

  double Sigma_g, Sigma_s, sigma_sz, rho_d;
  prfm_map_sample(p->x, &Sigma_g, &Sigma_s, &sigma_sz, &rho_d);
  if (Sigma_g <= 0.) {
    if (xp != NULL) {
      xp->sf_data.Sigma_g = 0.f;
      xp->sf_data.Sigma_s = 0.f;
      xp->sf_data.sigma_sz = 0.f;
      xp->sf_data.rho_d = 0.f;
      xp->sf_data.W = 0.f;
      xp->sf_data.sigma_eff = 0.f;
      xp->sf_data.rho_eq = 0.f;
      xp->sf_data.H_g = 0.f;
    }
    return 0;
  }

  prfm_int(Sigma_g * starform->Sigma_to_cgs, Sigma_s * starform->Sigma_to_cgs,
           rho_d * starform->rho_to_cgs, sigma_sz * starform->velocity_to_cgs,
           starform->c_d, &starform->cal, star_formation_prfm_Zprime(p, starform),
           starform->n_iter, state);

  if (xp != NULL) {
    xp->sf_data.Sigma_g = Sigma_g;
    xp->sf_data.Sigma_s = Sigma_s;
    xp->sf_data.sigma_sz = sigma_sz;
    xp->sf_data.rho_d = rho_d;
    xp->sf_data.W = state->W * starform->pressure_from_cgs;
    xp->sf_data.sigma_eff = state->sigma_eff / starform->velocity_to_cgs;
    xp->sf_data.rho_eq = state->W / (state->sigma_eff * state->sigma_eff) /
                         starform->rho_to_cgs;
    xp->sf_data.H_g = state->H_g * starform->length_from_cgs;
  }
  return 1;
}

/**
 * @brief Calculate if the gas particle satisfies the conditions for star
 * formation.
 *
 * @param p the gas particles.
 * @param xp the additional properties of the gas particles.
 * @param starform the star formation law properties to use.
 * @param phys_const the physical constants in internal units.
 * @param cosmo the cosmological parameters and properties.
 * @param hydro_props The properties of the hydro scheme.
 * @param us The internal system of units.
 * @param cooling The cooling data struct.
 * @param entropy_floor_props The entropy floor assumed in this run.
 */
INLINE static int star_formation_is_star_forming(
    const struct part *p, const struct xpart *xp,
    const struct star_formation *starform, const struct phys_const *phys_const,
    const struct cosmology *cosmo, const struct hydro_props *hydro_props,
    const struct unit_system *us, const struct cooling_function_data *cooling,
    const struct entropy_floor_properties *entropy_floor_props) {

  /* Density threshold */
  const double n_H =
      hydro_get_physical_density(p, cosmo) * starform->rho_to_n_H_cgs;
  if (starform->threshold == prfm_threshold_measured) {
    if (n_H < starform->n_th_H_p_cm3) return 0;
  } else {
    if (n_H < starform->n_floor_H_p_cm3) return 0;
    struct prfm_state state;
    if (!star_formation_prfm_state(p, starform, &state, NULL)) return 0;
    /* Sigma_g / (2 H_g) = W / sigma_eff^2 */
    const double rho_eq_cgs = state.W / (state.sigma_eff * state.sigma_eff);
    const double n_H_eq =
        rho_eq_cgs / starform->rho_to_cgs * starform->rho_to_n_H_cgs;
    if (n_H_eq < starform->n_th_H_p_cm3) return 0;
  }

  /* Cold enough? */
  const double T = cooling_get_temperature(phys_const, hydro_props, us, cosmo,
                                           cooling, p, xp);
  if (T < starform->T_max_K) return 1;

  /* Or on the effective EoS, within the margin */
  const double entropy = hydro_get_physical_entropy(p, xp, cosmo);
  const double entropy_eos = entropy_floor(p, cosmo, entropy_floor_props);
  return entropy < entropy_eos * starform->ten_to_entropy_margin_dex;
}

/**
 * @brief Compute the star-formation rate of a given particle and store
 * it into the #xpart: mdot = m_g / t_dep.
 *
 * @param p #part.
 * @param xp the #xpart.
 * @param starform the star formation law properties to use
 * @param phys_const the physical constants in internal units.
 * @param hydro_props The properties of the hydro scheme.
 * @param cosmo the cosmological parameters and properties.
 * @param dt_star The time-step of this particle.
 */
INLINE static void star_formation_compute_SFR(
    const struct part *p, struct xpart *xp,
    const struct star_formation *starform, const struct phys_const *phys_const,
    const struct hydro_props *hydro_props, const struct cosmology *cosmo,
    const double dt_star) {

  if (dt_star == 0.) {
    xp->sf_data.SFR = 0.f;
    return;
  }

  struct prfm_state state;
  if (!star_formation_prfm_state(p, starform, &state, xp)) {
    xp->sf_data.SFR = 0.f;
    return;
  }

  const double t_dep = state.t_dep * starform->time_from_cgs;
  xp->sf_data.SFR = hydro_get_mass(p) / t_dep;
}

/**
 * @brief Decides whether a particle should be converted into a
 * star or not: probability SFR dt / m_g, as the EAGLE module.
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param starform The properties of the star formation model.
 * @param e The #engine (for random numbers).
 * @param dt_star The time-step of this particle
 * @return 1 if a conversion should be done, 0 otherwise.
 */
INLINE static int star_formation_should_convert_to_star(
    const struct part *p, const struct xpart *xp,
    const struct star_formation *starform, const struct engine *e,
    const double dt_star) {

  const double prob = xp->sf_data.SFR * dt_star / hydro_get_mass(p);
  const double random_number =
      random_unit_interval(p->id, e->ti_current, random_number_star_formation);
  return (prob > random_number);
}

/**
 * @brief Returns the number of new star particles to create per SF event.
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param starform The properties of the star formation model.
 *
 * @return The number of extra star particles to generate per gas particles.
 *        (return 0 if the gas particle itself is to bhe converted)
 */
INLINE static int star_formation_number_spart_to_spawn(
    const struct part *p, const struct xpart *xp,
    const struct star_formation *starform) {
  return starform->num_stars_per_gas_particle - 1;
}

/**
 * @brief Returns the number of particles to convert per SF event.
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param starform The properties of the star formation model.
 *
 * @return The number of particles to generate per gas particles.
 *        (This has to be 0 or 1)
 */
INLINE static int star_formation_number_spart_to_convert(
    const struct part *p, const struct xpart *xp,
    const struct star_formation *starform) {
  return 1;
}

/**
 * @brief Update the SF properties of a particle that is not star forming.
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param e The #engine.
 * @param starform The properties of the star formation model.
 * @param with_cosmology Are we running with cosmology switched on?
 */
INLINE static void star_formation_update_part_not_SFR(
    struct part *p, struct xpart *xp, const struct engine *e,
    const struct star_formation *starform, const int with_cosmology) {

  /* The PRFM state for the snapshots, as for star-forming particles */
  struct prfm_state state;
  star_formation_prfm_state(p, starform, &state, xp);

  /* Check if it is the first time steps after star formation */
  if (xp->sf_data.SFR > 0.f) {

    /* Record the current time as an indicator of when this particle was last
       star-forming. */
    if (with_cosmology) {
      xp->sf_data.SFR = -e->cosmology->a;
    } else {
      xp->sf_data.SFR = -e->time;
    }
  }
}

/**
 * @brief Copies the properties of the gas particle over to the
 * star particle
 *
 * @param e The #engine
 * @param p the gas particles.
 * @param xp the additional properties of the gas particles.
 * @param sp the new created star particle with its properties.
 * @param starform the star formation law properties to use.
 * @param cosmo the cosmological parameters and properties.
 * @param with_cosmology if we run with cosmology.
 * @param phys_const the physical constants in internal units.
 * @param hydro_props The properties of the hydro scheme.
 * @param us The internal system of units.
 * @param cooling The cooling data struct.
 * @param chem_data The global properties of the chemistry scheme.
 * @param convert_part Did we convert a part (or spawned one)?
 * @param (return) displacement The 3D displacement vector of the star with
 * respect to the sink position.
 */
INLINE static void star_formation_copy_properties(
    const struct part *p, const struct xpart *xp, struct spart *sp,
    const struct engine *e, const struct star_formation *starform,
    const struct cosmology *cosmo, const int with_cosmology,
    const struct phys_const *phys_const, const struct hydro_props *hydro_props,
    const struct unit_system *us, const struct cooling_function_data *cooling,
    const struct chemistry_global_data *chem_data, const int convert_part,
    float displacement[3]) {

  /* Store the current mass */
  sp->mass = hydro_get_mass(p) / starform->num_stars_per_gas_particle;

  /* Store the current mass as the initial mass */
  sp->mass_init = hydro_get_mass(p) / starform->num_stars_per_gas_particle;

  /* Store either the birth_scale_factor or birth_time depending  */
  if (with_cosmology) {
    sp->birth_scale_factor = cosmo->a;
  } else {
    sp->birth_time = e->time;
  }

  /* Move over the splitting data */
  sp->split_data = xp->split_data;

  /* Store the chemistry struct in the star particle */
  sp->chemistry_data = p->chemistry_data;

  /* Store the tracers data */
  sp->tracers_data = xp->tracers_data;

#if defined(TRACERS_EAGLE)
  /* See star_formation/EAGLE/star_formation.h: clear the negative sentinel
   * of the trigger-recording mechanism. */
  for (int i = 0; i < num_snapshot_triggers_part; ++i)
    if (sp->tracers_data.averaged_SFR[i] < 0.f)
      sp->tracers_data.averaged_SFR[i] = 0.f;
#endif

  /* Store the birth density in the star particle */
  sp->birth_density = hydro_get_physical_density(p, cosmo);

  /* Store the birth temperature in the star particle */
  sp->birth_temperature = cooling_get_temperature(phys_const, hydro_props, us,
                                                  cosmo, cooling, p, xp);

  /* Flag that this particle has not done feedback yet */
  sp->f_E = -1.f;
  sp->number_of_SNII_events = 0;
  sp->last_enrichment_time = sp->birth_time;
  sp->count_since_last_enrichment = -1;
  sp->number_of_heating_events = 0.;

  /* If we are spawning more than 1 star per gas particle --> add some small
   * displacement */
  if (starform->num_stars_per_gas_particle > 1) {

    const float max_displacement = 0.1;
    const double delta_x =
        2.f * random_unit_interval(sp->id, e->ti_current,
                                   (enum random_number_type)0) -
        1.f;
    const double delta_y =
        2.f * random_unit_interval(sp->id, e->ti_current,
                                   (enum random_number_type)1) -
        1.f;
    const double delta_z =
        2.f * random_unit_interval(sp->id, e->ti_current,
                                   (enum random_number_type)2) -
        1.f;

    /* Update the displacement */
    displacement[0] = delta_x * max_displacement * p->h;
    displacement[1] = delta_y * max_displacement * p->h;
    displacement[2] = delta_z * max_displacement * p->h;

    /* Move the spart */
    sp->x[0] += displacement[0];
    sp->x[1] += displacement[1];
    sp->x[2] += displacement[2];

    /* Copy the position to the gpart */
    sp->gpart->x[0] = sp->x[0];
    sp->gpart->x[1] = sp->x[1];
    sp->gpart->x[2] = sp->x[2];
  }
}

/**
 * @brief initialization of the star formation law from the PRFMStarFormation
 * section.
 *
 * @param parameter_file The parsed parameter file
 * @param phys_const Physical constants in internal units
 * @param us The current internal system of units.
 * @param hydro_props The propertis of the hydro model.
 * @param cosmo The current cosmological model.
 * @param entropy_floor The properties of the entropy floor used in this
 * simulation.
 * @param starform the star formation law properties to initialize
 */
INLINE static void starformation_init_backend(
    struct swift_params *parameter_file, const struct phys_const *phys_const,
    const struct unit_system *us, const struct hydro_props *hydro_props,
    const struct cosmology *cosmo,
    const struct entropy_floor_properties *entropy_floor,
    struct star_formation *starform) {

  char cal[32] = "OK22";
  parser_get_opt_param_string(parameter_file, "PRFMStarFormation:calibration",
                              cal, "OK22");
  if (strcmp(cal, "OK22") == 0)
    starform->cal_choice = prfm_calibration_OK22;
  else if (strcmp(cal, "K24") == 0)
    starform->cal_choice = prfm_calibration_K24;
  else
    error("PRFMStarFormation:calibration must be OK22 or K24, got '%s'", cal);
  prfm_calibration_init(starform->cal_choice, &starform->cal);

  starform->c_d = parser_get_opt_param_double(
      parameter_file, "PRFMStarFormation:c_d", 16. * M_PI / 3.);
  starform->n_iter = parser_get_opt_param_int(
      parameter_file, "PRFMStarFormation:n_iterations", 5);
  starform->solar_metallicity_inv =
      1. / parser_get_opt_param_double(
               parameter_file, "PRFMStarFormation:solar_metallicity", 0.0134);

  char thr[32] = "measured";
  parser_get_opt_param_string(parameter_file, "PRFMStarFormation:threshold",
                              thr, "measured");
  if (strcmp(thr, "measured") == 0)
    starform->threshold = prfm_threshold_measured;
  else if (strcmp(thr, "equilibrium") == 0)
    starform->threshold = prfm_threshold_equilibrium;
  else
    error(
        "PRFMStarFormation:threshold must be 'measured' or 'equilibrium', "
        "got '%s'",
        thr);
  starform->n_th_H_p_cm3 = parser_get_opt_param_double(
      parameter_file, "PRFMStarFormation:threshold_n_H_p_cm3", 0.13);
  /* J26 p. 23 gives no value. At 1e7 Msun J26 Fig. 12 finds the measured
   * density about 1 dex below the equilibrium one, so the default floor sits
   * 1 dex below n_th = 0.13; it excludes gas far above the plane, whose
   * column passes the equilibrium cut. */
  starform->n_floor_H_p_cm3 = parser_get_opt_param_double(
      parameter_file, "PRFMStarFormation:equilibrium_min_measured_n_H_p_cm3",
      0.01);

  starform->T_max_K = parser_get_opt_param_double(
      parameter_file, "PRFMStarFormation:max_temperature_K", 1e5);
  starform->entropy_margin_dex = parser_get_opt_param_double(
      parameter_file, "PRFMStarFormation:EOS_entropy_margin_dex", 0.5);
  starform->ten_to_entropy_margin_dex = exp10(starform->entropy_margin_dex);

  /* Unit conversions */
  const double mass_cgs = units_cgs_conversion_factor(us, UNIT_CONV_MASS);
  const double length_cgs = units_cgs_conversion_factor(us, UNIT_CONV_LENGTH);
  starform->Sigma_to_cgs = mass_cgs / (length_cgs * length_cgs);
  starform->rho_to_cgs = units_cgs_conversion_factor(us, UNIT_CONV_DENSITY);
  starform->velocity_to_cgs =
      units_cgs_conversion_factor(us, UNIT_CONV_VELOCITY);
  starform->time_from_cgs =
      1. / units_cgs_conversion_factor(us, UNIT_CONV_TIME);
  starform->pressure_from_cgs =
      1. / units_cgs_conversion_factor(us, UNIT_CONV_PRESSURE);
  starform->length_from_cgs = 1. / length_cgs;
  starform->rho_to_n_H_cgs =
      starform->rho_to_cgs / (1.4 * phys_const->const_proton_mass * mass_cgs);

  prfm_map_params_init(&starform->map, parameter_file, phys_const);

  starform->num_stars_per_gas_particle = parser_get_opt_param_int(
      parameter_file, "PRFMStarFormation:num_of_stars_per_gas_particle", 1);
  if (starform->num_stars_per_gas_particle < 1)
    error(
        "The number of star particles formed per gas particle in a star "
        "formation event must be >0 !!");
}

/**
 * @brief Prints the used parameters of the star formation law
 *
 * @param starform the star formation law properties.
 * */
INLINE static void starformation_print_backend(
    const struct star_formation *starform) {

  message(
      "Star formation model is PRFM-int (Ostriker & Kim 2022; Jeffreson et "
      "al. 2026), calibration %s",
      starform->cal_choice == prfm_calibration_K24 ? "K24" : "OK22");
  message(
      "Upsilon = %.4g km/s (W/P0)^-%.3g Z'^%.3g, sigma_eff = %.4g km/s "
      "max(W/P0,1)^%.3g Z'^%.3g, c_d = %.4g, %d iterations",
      starform->cal.upsilon0 / 1e5, starform->cal.alpha, starform->cal.upsilon_Z,
      starform->cal.sigma0 / 1e5, starform->cal.beta, starform->cal.sigma_Z,
      starform->c_d, starform->n_iter);
  message(
      "Threshold n_H >= %.3g cm^-3 (%s density), and T < %.3g K or entropy "
      "within %.3g dex of the floor",
      starform->n_th_H_p_cm3,
      starform->threshold == prfm_threshold_measured ? "measured"
                                                     : "equilibrium",
      starform->T_max_K, starform->entropy_margin_dex);
  if (starform->threshold == prfm_threshold_equilibrium)
    message("Equilibrium threshold also requires a measured n_H >= %.3g cm^-3",
            starform->n_floor_H_p_cm3);
  prfm_map_params_print(&starform->map);
  message("Will form %d stars per gas particle.",
          starform->num_stars_per_gas_particle);
}

/**
 * @brief Return the star formation rate of a particle.
 *
 * @param p The particle.
 * @param xp The extended data of the particle.
 */
INLINE static float star_formation_get_SFR(const struct part *p,
                                           const struct xpart *xp) {
  if (xp->sf_data.SFR <= 0.)
    return 0.f;
  else
    return xp->sf_data.SFR;
}

/**
 * @brief Finishes the density calculation.
 *
 * Nothing to do here.
 *
 * @param p The particle to act upon
 * @param xp The extra particle to act upon
 * @param cd The global star_formation information.
 * @param cosmo The current cosmological model.
 */
__attribute__((always_inline)) INLINE static void star_formation_end_density(
    struct part *p, struct xpart *xp, const struct star_formation *cd,
    const struct cosmology *cosmo) {}

/**
 * @brief Sets all particle fields to sensible values when the #part has 0 ngbs.
 *
 * Nothing to do here.
 *
 * @param p The particle to act upon
 * @param xp The extended particle data to act upon
 * @param cd #star_formation containing star_formation informations.
 * @param cosmo The current cosmological model.
 */
__attribute__((always_inline)) INLINE static void
star_formation_part_has_no_neighbours(struct part *p, struct xpart *xp,
                                      const struct star_formation *cd,
                                      const struct cosmology *cosmo) {}

/**
 * @brief Sets the star_formation properties of the (x-)particles to a valid
 * state to start the density loop.
 *
 * Nothing to do here.
 *
 * @param data The global star_formation information used for this run.
 * @param p Pointer to the particle data.
 */
__attribute__((always_inline)) INLINE static void star_formation_init_part(
    struct part *p, const struct star_formation *data) {}

/**
 * @brief Sets the star_formation properties of the (x-)particles to a valid
 * start state at the beginning of the simulation after the ICs have been read:
 * no PRFM state yet.
 *
 * @param phys_const The physical constant in internal units.
 * @param us The unit system.
 * @param cosmo The current cosmological model.
 * @param data The global star_formation information used for this run.
 * @param p Pointer to the particle data.
 * @param xp Pointer to the extended particle data.
 */
__attribute__((always_inline)) INLINE static void
star_formation_first_init_part(const struct phys_const *phys_const,
                               const struct unit_system *us,
                               const struct cosmology *cosmo,
                               const struct star_formation *data,
                               const struct part *p, struct xpart *xp) {
  xp->sf_data.Sigma_g = 0.f;
  xp->sf_data.Sigma_s = 0.f;
  xp->sf_data.sigma_sz = 0.f;
  xp->sf_data.rho_d = 0.f;
  xp->sf_data.W = 0.f;
  xp->sf_data.sigma_eff = 0.f;
  xp->sf_data.rho_eq = 0.f;
  xp->sf_data.H_g = 0.f;
}

/**
 * @brief Split the star formation content of a particle into n pieces
 *
 * @param p The #part.
 * @param xp The #xpart.
 * @param n The number of pieces to split into.
 */
__attribute__((always_inline)) INLINE static void star_formation_split_part(
    struct part *p, struct xpart *xp, const double n) {

  if (xp->sf_data.SFR > 0.) xp->sf_data.SFR /= n;
}

/**
 * @brief Deal with the case where no spart are available for star formation.
 *
 * @param e The #engine.
 * @param p The #part.
 * @param xp The #xpart.
 */
__attribute__((always_inline)) INLINE static void
star_formation_no_spart_available(const struct engine *e, const struct part *p,
                                  const struct xpart *xp) {
  /* Nothing to do, we just skip it and deal with it next step */
}

/**
 * @brief Compute some information for the star formation model based
 * on all the particles that were read in.
 *
 * Nothing to do here.
 *
 * @param star_form The #star_formation structure.
 * @param e The #engine.
 */
__attribute__((always_inline)) INLINE static void
star_formation_first_init_stats(struct star_formation *star_form,
                                const struct engine *e) {}

#endif /* SWIFT_PRFM_STAR_FORMATION_H */
