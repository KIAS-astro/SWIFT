/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Jeong-Gyu Kim (omyojj@gmail.com)
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
 ******************************************************************************/
#ifndef SWIFT_POTENTIAL_NFW_DK_H
#define SWIFT_POTENTIAL_NFW_DK_H

/* Config parameters. */
#include <config.h>

/* Some standard headers. */
#include <float.h>
#include <math.h>

/* Local includes. */
#include "error.h"
#include "gravity.h"
#include "parser.h"
#include "part.h"
#include "physical_constants.h"
#include "space.h"
#include "units.h"

/**
 * @brief External Potential Properties - NFW halo with the Diemer & Kravtsov
 * (2014) outer profile.
 *
 *   rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
 *            + rho_m [b_e (r / 5 R_200m)^(-s_e) + f_mean]
 *
 * The second term is eq. 4 of Diemer & Kravtsov (2014) with rho_m = Omega_m
 * rho_crit; f_mean = 1 keeps their mean-density term. The transition
 * function f_trans of their eq. 6 is not applied to the NFW part. This is the
 * static halo of the isolated galaxy of Smith et al. (2024, MNRAS 527, 1216,
 * Table 1). M_200 and R_200 refer to the critical density, R_200m to the mean
 * matter density and is found from the NFW mass alone at start-up.
 */
struct external_potential {

  /*! Position of the centre of potential */
  double x[3];

  /*! The scale radius of the NFW potential */
  double r_s;

  /*! The concentration parameter */
  double c_200;

  /*! The mass at R200 */
  double M_200;

  /*! Radius within which the mean density is 200 rho_crit */
  double R_200;

  /*! Radius within which the NFW mean density is 200 rho_m */
  double R_200m;

  /*! Mean matter density Omega_m rho_crit */
  double rho_m;

  /*! Outer power-law slope s_e and normalisation b_e */
  double s_e, b_e;

  /*! Weight of the mean-density term (1: DK14 eq. 4; 0: power law only) */
  double f_mean;

  /*! rho_m b_e (5 R_200m)^s_e: rho_outer = A r^-s_e + f_mean rho_m */
  double A_outer;

  /*! M_200 / [ln(1 + c) - c / (1 + c)] */
  double M_s;

  /*! Time-step condition pre_factor, this factor is used to multiply times the
   * orbital time, so in the case of 0.01 we take 1% of the orbital time as
   * the time integration steps */
  double timestep_mult;

  /*! Minimum time step based on the orbital time at the softening times
   * the timestep_mult */
  double mintime;

  /*! The softening length */
  double eps;
};

/**
 * @brief Mass enclosed within radius r (internal units).
 */
__attribute__((always_inline)) INLINE static double nfw_dk_enclosed_mass(
    const struct external_potential *restrict potential, const double r) {

  const double x = r / potential->r_s;
  const double m_nfw = potential->M_s * (log(1. + x) - x / (1. + x));
  const double s = potential->s_e;
  const double m_out =
      4. * M_PI *
      (potential->A_outer * pow(r, 3. - s) / (3. - s) +
       potential->f_mean * potential->rho_m * r * r * r / 3.);
  return m_nfw + m_out;
}

/**
 * @brief Density of the halo at radius r (internal units).
 */
__attribute__((always_inline)) INLINE static double nfw_dk_density(
    const struct external_potential *restrict potential, const double r) {

  const double x = r / potential->r_s;
  const double rs3 = potential->r_s * potential->r_s * potential->r_s;
  const double rho_nfw =
      potential->M_s / (4. * M_PI * rs3) / (x * (1. + x) * (1. + x));
  return rho_nfw + potential->A_outer * pow(r, -potential->s_e) +
         potential->f_mean * potential->rho_m;
}

/**
 * @brief Gravitational potential per unit G at radius r, zero at r = 0 for
 * the outer terms (they have no finite value at infinity for s_e < 2).
 */
__attribute__((always_inline)) INLINE static double nfw_dk_potential_over_G(
    const struct external_potential *restrict potential, const double r) {

  const double s = potential->s_e;
  const double phi_nfw = -potential->M_s * log(1. + r / potential->r_s) / r;
  const double phi_pl =
      4. * M_PI * potential->A_outer * pow(r, 2. - s) / ((3. - s) * (2. - s));
  const double phi_mean =
      2. * M_PI / 3. * potential->f_mean * potential->rho_m * r * r;
  return phi_nfw + phi_pl + phi_mean;
}

/**
 * @brief Computes the time-step due to the acceleration from the potential:
 * timestep_mult times the circular orbital period at the softened radius.
 *
 * @param time The current time.
 * @param potential The #external_potential used in the run.
 * @param phys_const The physical constants in internal units.
 * @param g Pointer to the g-particle data.
 */
__attribute__((always_inline)) INLINE static float external_gravity_timestep(
    double time, const struct external_potential *restrict potential,
    const struct phys_const *restrict phys_const,
    const struct gpart *restrict g) {

  const double dx = g->x[0] - potential->x[0];
  const double dy = g->x[1] - potential->x[1];
  const double dz = g->x[2] - potential->x[2];

  const double r =
      sqrt(dx * dx + dy * dy + dz * dz + potential->eps * potential->eps);
  const double mr = nfw_dk_enclosed_mass(potential, r);
  const double period =
      2. * M_PI * r * sqrt(r / (phys_const->const_newton_G * mr));

  return max(potential->timestep_mult * period, potential->mintime);
}

/**
 * @brief Computes the gravitational acceleration from the halo. As in the
 * other SWIFT potentials, a_grav is stored per unit G.
 *
 * @param time The current time in internal units.
 * @param potential The properties of the potential.
 * @param phys_const The physical constants in internal units.
 * @param g Pointer to the g-particle data.
 */
__attribute__((always_inline)) INLINE static void external_gravity_acceleration(
    double time, const struct external_potential *restrict potential,
    const struct phys_const *restrict phys_const, struct gpart *restrict g) {

  const double dx = g->x[0] - potential->x[0];
  const double dy = g->x[1] - potential->x[1];
  const double dz = g->x[2] - potential->x[2];

  const double r =
      sqrt(dx * dx + dy * dy + dz * dz + potential->eps * potential->eps);
  const double r_inv = 1. / r;
  const double acc = -nfw_dk_enclosed_mass(potential, r) * r_inv * r_inv * r_inv;

  g->a_grav[0] += acc * dx;
  g->a_grav[1] += acc * dy;
  g->a_grav[2] += acc * dz;
  gravity_add_comoving_potential(g, nfw_dk_potential_over_G(potential, r));
}

/**
 * @brief Computes the gravitational potential energy per unit mass of a
 * particle in the halo.
 *
 * @param time The current time.
 * @param potential The #external_potential used in the run.
 * @param phys_const Physical constants in internal units.
 * @param g Pointer to the particle data.
 */
__attribute__((always_inline)) INLINE static float
external_gravity_get_potential_energy(
    double time, const struct external_potential *potential,
    const struct phys_const *const phys_const, const struct gpart *g) {

  const double dx = g->x[0] - potential->x[0];
  const double dy = g->x[1] - potential->x[1];
  const double dz = g->x[2] - potential->x[2];

  const double r =
      sqrt(dx * dx + dy * dy + dz * dz + potential->eps * potential->eps);
  return phys_const->const_newton_G * nfw_dk_potential_over_G(potential, r);
}

/**
 * @brief Initialises the external potential properties in the internal
 * system of units.
 *
 * @param parameter_file The parsed parameter file
 * @param phys_const Physical constants in internal units
 * @param us The current internal system of units
 * @param s The #space we run in.
 * @param potential The external potential properties to initialize
 */
static INLINE void potential_init_backend(
    struct swift_params *parameter_file, const struct phys_const *phys_const,
    const struct unit_system *us, const struct space *s,
    struct external_potential *potential) {

  parser_get_param_double_array(parameter_file, "NFWDKPotential:position", 3,
                                potential->x);
  const int useabspos =
      parser_get_param_int(parameter_file, "NFWDKPotential:useabspos");
  if (!useabspos) {
    potential->x[0] += s->dim[0] / 2.;
    potential->x[1] += s->dim[1] / 2.;
    potential->x[2] += s->dim[2] / 2.;
  }

  potential->timestep_mult =
      parser_get_param_double(parameter_file, "NFWDKPotential:timestep_mult");
  potential->c_200 =
      parser_get_param_double(parameter_file, "NFWDKPotential:concentration");
  potential->M_200 =
      parser_get_param_double(parameter_file, "NFWDKPotential:M_200");
  potential->eps =
      parser_get_param_double(parameter_file, "NFWDKPotential:epsilon");
  const double h = parser_get_param_double(parameter_file, "NFWDKPotential:h");
  const double Omega_m =
      parser_get_param_double(parameter_file, "NFWDKPotential:Omega_m");
  potential->s_e =
      parser_get_opt_param_double(parameter_file, "NFWDKPotential:s_e", 1.5);
  potential->b_e =
      parser_get_opt_param_double(parameter_file, "NFWDKPotential:b_e", 1.0);
  potential->f_mean = parser_get_opt_param_double(
      parameter_file, "NFWDKPotential:mean_density_weight", 1.0);
  if (potential->s_e <= 0. || potential->s_e >= 2.)
    error("NFWDKPotential:s_e must lie in (0, 2), got %g", potential->s_e);

  const double G_newton = phys_const->const_newton_G;
  const double H0 = h * phys_const->const_reduced_hubble;
  const double rho_crit = 3. * H0 * H0 / (8. * M_PI * G_newton);

  potential->R_200 = cbrt(10. * potential->M_200 * G_newton * H0) / (10 * H0);
  potential->r_s = potential->R_200 / potential->c_200;
  potential->M_s =
      potential->M_200 /
      (log(1. + potential->c_200) - potential->c_200 / (1. + potential->c_200));
  potential->rho_m = Omega_m * rho_crit;

  /* R_200m: NFW mass = 200 rho_m (4 pi / 3) R^3, by bisection; the NFW mean
   * density falls monotonically, so the root is unique. */
  double lo = potential->R_200, hi = 100. * potential->R_200;
  for (int i = 0; i < 200; i++) {
    const double mid = 0.5 * (lo + hi);
    const double x = mid / potential->r_s;
    const double m = potential->M_s * (log(1. + x) - x / (1. + x));
    if (m > 200. * potential->rho_m * 4. * M_PI / 3. * mid * mid * mid)
      lo = mid;
    else
      hi = mid;
  }
  potential->R_200m = 0.5 * (lo + hi);
  potential->A_outer = potential->rho_m * potential->b_e *
                       pow(5. * potential->R_200m, potential->s_e);

  const double eps = potential->eps;
  potential->mintime =
      potential->timestep_mult * 2. * M_PI * eps *
      sqrt(eps / (G_newton * nfw_dk_enclosed_mass(potential, eps)));
}

/**
 * @brief Prints the properties of the external potential to stdout.
 *
 * @param  potential The external potential properties.
 */
static INLINE void potential_print_backend(
    const struct external_potential *potential) {

  message(
      "External potential is 'NFW + Diemer-Kravtsov outer profile' at (x,y,z) "
      "= (%e, %e, %e), timestep multiplier = %e, mintime = %e",
      potential->x[0], potential->x[1], potential->x[2],
      potential->timestep_mult, potential->mintime);
  message(
      "M200 = %e, R200 = %e, c = %e, r_s = %e; R200m = %e, rho_m = %e, s_e = "
      "%g, b_e = %g, mean-density weight %g, softening %e",
      potential->M_200, potential->R_200, potential->c_200, potential->r_s,
      potential->R_200m, potential->rho_m, potential->s_e, potential->b_e,
      potential->f_mean, potential->eps);
}

#endif /* SWIFT_POTENTIAL_NFW_DK_H */
