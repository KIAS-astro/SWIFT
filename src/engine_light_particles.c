/*******************************************************************************
 * This file is part of SWIFT.
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

/**
 * @file src/engine_light_particles.c
 * @brief Light wind particles (Wind:mode light): spawning from the wind
 * reservoirs and merging by the refinement clock, both at tree rebuilds
 * (light-particles-design.md Sec. 8).
 *
 * Both run after every particle has been drifted to the current time and
 * before engine_rebuild(), node-local like engine_split_gas_particles():
 * merging first, while the cells of the old tree still index the particle
 * arrays; spawning second, since it may reallocate them. New particles are
 * appended to the arrays and sorted into cells by the rebuild that follows;
 * merged-away particles are inhibited and removed by it.
 */

/* Config parameters. */
#include <config.h>

/* This object's header. */
#include "engine.h"

#ifdef WIND_TIGRESS

/* Local headers */
#include "active.h"
#include "black_holes_struct.h"
#include "cell.h"
#include "chemistry.h"
#include "cooling.h"
#include "hydro.h"
#include "light_physics.h"
#include "periodic.h"
#include "random.h"
#include "sink_struct.h"
#include "star_formation.h"
#include "threadpool.h"
#include "tools.h"
#include "tracers.h"
#include "wind.h"

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <math.h>
#include <string.h>

/**
 * @brief Mix the chemistry of b into a: mass fractions by mass, masses
 * summed.
 */
static void light_merge_chemistry(struct part *a, const double ma,
                                  const struct part *b, const double mb) {
#if defined(CHEMISTRY_EAGLE)
  const float m_SNIa = a->chemistry_data.mass_from_SNIa +
                       b->chemistry_data.mass_from_SNIa;
  const float m_SNII = a->chemistry_data.mass_from_SNII +
                       b->chemistry_data.mass_from_SNII;
  const float m_AGB =
      a->chemistry_data.mass_from_AGB + b->chemistry_data.mass_from_AGB;

  /* Every other field of the EAGLE gas chemistry is a mass fraction */
  float *fa = (float *)&a->chemistry_data;
  const float *fb = (const float *)&b->chemistry_data;
  const size_t n = sizeof(struct chemistry_part_data) / sizeof(float);
  const double M = ma + mb;
  for (size_t i = 0; i < n; i++) fa[i] = (ma * fa[i] + mb * fb[i]) / M;

  a->chemistry_data.mass_from_SNIa = m_SNIa;
  a->chemistry_data.mass_from_SNII = m_SNII;
  a->chemistry_data.mass_from_AGB = m_AGB;
#elif defined(CHEMISTRY_NONE)
#else
#error "Light-particle merging needs EAGLE or no chemistry."
#endif
}

/**
 * @brief Combine the hydrodynamical updates of a and b that are still to be
 * applied in their next kicks, so that the momentum and energy they carry
 * are kept.
 *
 * GIZMO-MFM: accumulated fluxes are extensive and add. SPH: the accelerations
 * and the internal-energy rates are per unit mass and are mass-weighted.
 */
static void light_merge_hydro_rates(struct part *a, const double ma,
                                    const struct part *b, const double mb,
                                    const struct cosmology *cosmo) {
#if defined(GIZMO_MFM_SPH)
  for (int k = 0; k < 3; k++) a->flux.momentum[k] += b->flux.momentum[k];
  a->flux.energy += b->flux.energy;
#else
  const double M = ma + mb;
  for (int k = 0; k < 3; k++)
    a->a_hydro[k] = (ma * a->a_hydro[k] + mb * b->a_hydro[k]) / M;
  const double du_dt =
      (ma * hydro_get_physical_internal_energy_dt(a, cosmo) +
       mb * hydro_get_physical_internal_energy_dt(b, cosmo)) /
      M;
  hydro_set_physical_internal_energy_dt(a, cosmo, du_dt);
#endif
}

/**
 * @brief Zero the hydrodynamical updates a spawned particle copied from its
 * parent: they belong to the parent.
 */
static void light_reset_hydro_rates(struct part *p) {
#if defined(GIZMO_MFM_SPH)
  hydro_part_reset_hydro_fluxes(p);
  /* Inactive in the flux exchange until its own time-step is set: a clone of
   * a parent in mid-step would otherwise carry the parent's positive dt */
  p->flux.dt = -1.0f;
#endif
  hydro_reset_acceleration(p);
}

/**
 * @brief Separation b - a on the nearest periodic image.
 */
static void light_separation(const struct space *s, const double *xa,
                             const double *xb, double dx[3]) {
  for (int k = 0; k < 3; k++) {
    dx[k] = xb[k] - xa[k];
    if (s->periodic) dx[k] = nearest(dx[k], s->dim[k]);
  }
}

/**
 * @brief Merge particle b into particle a (light-particles-design.md
 * Sec. 4.2): mass, momentum, metal mass and total energy conserved, the
 * relative kinetic energy turned into heat, centre-of-mass position,
 * mass-weighted dye. b is removed.
 */
static void light_merge_pair(struct engine *e, struct cell *c, struct part *a,
                             struct xpart *xa, struct part *b,
                             struct xpart *xb) {

  const struct wind_props *props = e->wind_props;
  const struct cosmology *cosmo = e->cosmology;
  const struct space *s = e->s;
  const double time = e->time;

  const double ma = hydro_get_mass(a);
  const double mb = hydro_get_mass(b);

  double dx[3];
  light_separation(s, a->x, b->x, dx);
  double x[3] = {a->x[0], a->x[1], a->x[2]};
  const double x_b[3] = {a->x[0] + dx[0], a->x[1] + dx[1], a->x[2] + dx[2]};
  double v[3] = {xa->v_full[0], xa->v_full[1], xa->v_full[2]};
  const double v_b[3] = {xb->v_full[0], xb->v_full[1], xb->v_full[2]};
  double u = hydro_get_physical_internal_energy(a, xa, cosmo);
  const double u_b = hydro_get_physical_internal_energy(b, xb, cosmo);

  const double f_a = wind_light_dye(a, props, time);
  const double f_b = wind_light_dye(b, props, time);

  double M = ma;
  const double heat = light_merge_state(&M, x, v, &u, mb, x_b, v_b, u_b);

  /* Drifted SPH velocity, mass-weighted like v_full */
  float v_pred[3];
  for (int k = 0; k < 3; k++)
    v_pred[k] = (ma * a->v[k] + mb * b->v[k]) / M;

  light_merge_chemistry(a, ma, b, mb);
  light_merge_hydro_rates(a, ma, b, mb, cosmo);

  /* Mass first, then velocity, then energy: the order the GIZMO setters
   * need (hydro_set_mass keeps v and u; the v_full change reaches the MFM
   * momentum at the next kick) */
  hydro_set_mass(a, M);
  for (int k = 0; k < 3; k++) {
    a->x[k] = s->periodic ? box_wrap(x[k], 0.0, s->dim[k]) : x[k];
    xa->v_full[k] = v[k];
    a->v[k] = v_pred[k];
  }
  if (a->gpart != NULL) {
    a->gpart->mass = M;
    for (int k = 0; k < 3; k++) {
      a->gpart->x[k] = a->x[k];
      a->gpart->v_full[k] = xa->v_full[k];
    }
  }
  hydro_set_physical_internal_energy(a, xa, cosmo, u);
  /* An active particle had its density reset by the drift that precedes the
   * rebuild; its drifted energy is reset from the full one at its next kick,
   * and its pressure is recomputed by this step's density loop */
  if (a->rho > 0.f) hydro_set_drifted_physical_internal_energy(a, cosmo, NULL, u);

  /* Reservoirs and wind state */
  a->wind_data.m_pending += b->wind_data.m_pending;
  a->wind_data.E_pending += b->wind_data.E_pending;
  a->wind_data.dye = (ma * f_a + mb * f_b) / M;
  a->wind_data.dye_time = time;
  a->wind_data.light = a->wind_data.light || b->wind_data.light;
  if (a->wind_data.dye <= props->light_fw_ref && M >= 0.5 * props->light_m_g)
    a->wind_data.light = 0;
  a->wind_data.light_merges++;

  atomic_add_d(&wind_global_stats.n_merge, 1.);
  atomic_add_d(&wind_global_stats.m_merge, mb);
  atomic_add_d(&wind_global_stats.E_merge_heat, heat);

  b->wind_data.m_pending = 0.;
  b->wind_data.E_pending = 0.;
  cell_remove_part(e, c, b, xb);
}

/**
 * @brief Is p a coupled light particle below half its target mass?
 */
static int light_wants_merge(const struct part *p, const struct engine *e,
                             const struct wind_props *props,
                             const double time) {
  if (p->time_bin == time_bin_inhibited || p->wind_data.decoupled ||
      !p->wind_data.light)
    return 0;
  return hydro_get_mass(p) < 0.5 * wind_light_target_mass(p, props, time);
}

/**
 * @brief Merge the light particles of one leaf cell.
 *
 * A coupled light particle below half its target mass merges with its
 * nearest neighbour inside its kernel support that is also a light
 * particle below half its target. With its clock run out (f_w <= f_w,ref)
 * and no such partner, it merges with the nearest coupled gas particle that
 * stays below 3/2 m_g.
 * Partners may be in mid-step: the removed particle's pending hydro update
 * (SPH acceleration and du/dt, mass-weighted; GIZMO fluxes, summed) passes to
 * the survivor. A rebuild usually falls on a step where only the shortest
 * time-bins are active, so requiring active partners would leave halo light
 * particles, on long steps, almost never merged. The survivor is the partner
 * on the shorter time-bin (lower id on a tie), whose integration state is
 * kept.
 */
static void light_merge_leaf(struct engine *e, struct cell *c) {

  const struct wind_props *props = e->wind_props;
  const struct space *s = e->s;
  const double time = e->time;
  const int count = c->hydro.count;
  struct part *parts = c->hydro.parts;
  struct xpart *xparts = c->hydro.xparts;

  for (int i = 0; i < count; i++) {
    struct part *p = &parts[i];
    if (p->time_bin == time_bin_inhibited || p->wind_data.decoupled ||
        !p->wind_data.light)
      continue;

    /* Clock run out at a mass of at least m_g / 2: an ordinary particle */
    const double f_w = wind_light_dye(p, props, time);
    const double m = hydro_get_mass(p);
    if (f_w <= props->light_fw_ref && m >= 0.5 * props->light_m_g) {
      p->wind_data.light = 0;
      continue;
    }
    if (!light_wants_merge(p, e, props, time)) continue;

    const double r_max = kernel_gamma * p->h;
    double r2_light = r_max * r_max, r2_any = r_max * r_max;
    int j_light = -1, j_any = -1;
    for (int j = 0; j < count; j++) {
      if (j == i) continue;
      const struct part *q = &parts[j];
      if (q->time_bin == time_bin_inhibited || q->wind_data.decoupled) continue;
      double dx[3];
      light_separation(s, p->x, q->x, dx);
      const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
      if (r2 < r2_any && hydro_get_mass(q) + m <= 1.5 * props->light_m_g) {
        r2_any = r2;
        j_any = j;
      }
      if (r2 < r2_light && light_wants_merge(q, e, props, time)) {
        r2_light = r2;
        j_light = j;
      }
    }

    int j = j_light;
    if (j < 0 && f_w <= props->light_fw_ref) j = j_any;
    if (j < 0) continue;

    struct part *q = &parts[j];
    const int p_survives =
        p->time_bin < q->time_bin || (p->time_bin == q->time_bin && p->id < q->id);
    if (p_survives)
      light_merge_pair(e, c, p, &xparts[i], q, &xparts[j]);
    else
      light_merge_pair(e, c, q, &xparts[j], p, &xparts[i]);
  }
}

/**
 * @brief Recurse to the leaves of a cell.
 */
static void light_merge_recurse(struct engine *e, struct cell *c) {
  if (c->hydro.count == 0) return;
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) light_merge_recurse(e, c->progeny[k]);
  } else {
    light_merge_leaf(e, c);
  }
}

/**
 * @brief Mapper over the local top-level cells.
 */
static void light_merge_mapper(void *map_data, int num_elements,
                               void *extra_data) {
  struct engine *e = (struct engine *)extra_data;
  int *local_cells = (int *)map_data;
  for (int ind = 0; ind < num_elements; ind++) {
    struct cell *c = &e->s->cells_top[local_cells[ind]];
    if (c->nodeID != e->nodeID) continue;
    light_merge_recurse(e, c);
  }
}

/**
 * @brief Merge light particles by their refinement clock (node-local).
 */
void engine_light_merge(struct engine *e) {
  if (!e->wind_props->light) return;
  const ticks tic = getticks();
  struct space *s = e->s;
  threadpool_map(&e->threadpool, light_merge_mapper, s->local_cells_top,
                 s->nr_local_cells, sizeof(int), threadpool_auto_chunk_size,
                 e);
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/**
 * @brief Grow the part, xpart and gpart arrays to hold n_new more particles.
 */
static void light_grow_arrays(struct engine *e, const size_t n_new,
                              const int with_gravity) {
  struct space *s = e->s;

  if (s->nr_parts + n_new > s->size_parts) {
    s->size_parts = engine_parts_size_grow * (s->nr_parts + n_new);
    struct part *parts_new = NULL;
    struct xpart *xparts_new = NULL;
    if (swift_memalign("parts", (void **)&parts_new, part_align,
                       sizeof(struct part) * s->size_parts) != 0 ||
        swift_memalign("xparts", (void **)&xparts_new, xpart_align,
                       sizeof(struct xpart) * s->size_parts) != 0)
      error("Failed to allocate new part data.");
    memcpy(parts_new, s->parts, sizeof(struct part) * s->nr_parts);
    memcpy(xparts_new, s->xparts, sizeof(struct xpart) * s->nr_parts);
    swift_free("parts", s->parts);
    swift_free("xparts", s->xparts);
    s->parts = parts_new;
    s->xparts = xparts_new;
  }

  if (with_gravity && s->nr_gparts + n_new > s->size_gparts) {
    s->size_gparts = engine_parts_size_grow * (s->nr_gparts + n_new);
    struct gpart *gparts_new = NULL;
    if (swift_memalign("gparts", (void **)&gparts_new, gpart_align,
                       sizeof(struct gpart) * s->size_gparts) != 0)
      error("Failed to allocate new gpart data.");
    memcpy(gparts_new, s->gparts, sizeof(struct gpart) * s->nr_gparts);
    swift_free("gparts", s->gparts);
    part_relink_all_parts_to_gparts(gparts_new, s->nr_gparts, s->parts,
                                    s->sinks, s->sparts, s->bparts,
                                    &e->threadpool);
    s->gparts = gparts_new;
  }
}

/**
 * @brief Append a copy of parts[i_src] (with its gpart) to the arrays and
 * return its index. Arrays must have room.
 */
static size_t light_append_copy(struct space *s, const size_t i_src,
                                const long long id, const int with_gravity) {
  const size_t k = s->nr_parts++;
  memcpy(&s->parts[k], &s->parts[i_src], sizeof(struct part));
  memcpy(&s->xparts[k], &s->xparts[i_src], sizeof(struct xpart));
  s->parts[k].id = id;
  s->parts[k].gpart = NULL;
  if (with_gravity && s->parts[i_src].gpart != NULL) {
    const size_t kg = s->nr_gparts++;
    memcpy(&s->gparts[kg], s->parts[i_src].gpart, sizeof(struct gpart));
    s->parts[k].gpart = &s->gparts[kg];
    s->gparts[kg].id_or_neg_offset = -(long long)k;
  }
  return k;
}

/**
 * @brief Set the mass of a spawned particle copied from a particle of mass
 * m_src, scaling the extensive fields of the subgrid modules.
 */
static void light_set_spawned_mass(struct part *p, struct xpart *xp,
                                   const double m_src, const double m) {
  const double factor = m_src / m;
  hydro_set_mass(p, m);
  if (p->gpart != NULL) p->gpart->mass = m;
  chemistry_split_part(p, factor);
  cooling_split_part(p, xp, factor);
  star_formation_split_part(p, xp, factor);
  tracers_split_part(p, xp, factor);
  black_holes_mark_part_as_not_swallowed(&p->black_holes_data);
  sink_mark_part_as_not_swallowed(&p->sink_data);
  light_reset_hydro_rates(p);
}

/**
 * @brief Move a particle (and its gpart) by dx, wrapping periodically.
 */
static void light_move(const struct space *s, struct part *p,
                       const double dx[3]) {
  for (int k = 0; k < 3; k++) {
    double x = p->x[k] + dx[k];
    if (s->periodic) x = box_wrap(x, 0.0, s->dim[k]);
    p->x[k] = x;
    if (p->gpart != NULL) p->gpart->x[k] = x;
  }
}

/**
 * @brief Spawn light particles from the reservoirs and from the optional
 * central source (node-local).
 */
void engine_light_spawn(struct engine *e) {

  struct wind_props *props = e->wind_props;
  if (!props->light) return;
  const ticks tic = getticks();

  struct space *s = e->s;
  const struct cosmology *cosmo = e->cosmology;
  const int with_gravity = (e->policy & engine_policy_self_gravity) ||
                           (e->policy & engine_policy_external_gravity);
  const double m_l = props->light_m_l;
  const double time = e->time;
  const size_t nr_parts_old = s->nr_parts;

  /* Count the particles to create */
  long long n_new = 0;
  for (size_t i = 0; i < nr_parts_old; i++) {
    const struct part *p = &s->parts[i];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created || p->wind_data.decoupled)
      continue;
    n_new += light_spawn_count(
        hydro_get_mass(p), p->wind_data.m_pending, m_l,
        random_unit_interval(p->id, e->ti_current, random_number_light_spawn));
  }

  /* Central source, on rank 0: the mass injected since the last rebuild */
  long long n_source = 0;
  size_t i_template = nr_parts_old;
  if (props->source_Mdot > 0. && e->nodeID == 0) {
    const double t_prev = props->light_last_rebuild > -FLT_MAX
                              ? props->light_last_rebuild
                              : e->time_begin;
    props->source_m_carry += props->source_Mdot * (time - t_prev);
    n_source = (long long)floor(props->source_m_carry / m_l);
    for (size_t i = 0; i < nr_parts_old && n_source > 0; i++) {
      const struct part *p = &s->parts[i];
      if (p->time_bin != time_bin_inhibited &&
          p->time_bin != time_bin_not_created && !p->wind_data.decoupled) {
        i_template = i;
        break;
      }
    }
    if (n_source > 0 && i_template == nr_parts_old)
      error("No gas particle on rank 0 to copy for the light source");
  }
  props->light_last_rebuild = time;

  /* Unique ids above every gas id, as engine_split_gas_particles assigns
   * them (factor 2 keeps the parity of the ids) */
  long long expected_count_id = 2 * (n_new + n_source);
  long long offset_id = 0;
#ifdef WITH_MPI
  MPI_Exscan(&expected_count_id, &offset_id, 1, MPI_LONG_LONG_INT, MPI_SUM,
             MPI_COMM_WORLD);
#endif
  offset_id += e->max_parts_id + 1;
  e->max_parts_id = offset_id + expected_count_id;
#ifdef WITH_MPI
  MPI_Bcast(&e->max_parts_id, 1, MPI_LONG_LONG_INT, e->nr_nodes - 1,
            MPI_COMM_WORLD);
#endif

  /* No early return when nothing is spawned: every reservoir must still be
   * emptied by this draw */
  light_grow_arrays(e, n_new + n_source, with_gravity);
  long long next_id = offset_id;

  /* Clones of the parents */
  for (size_t i = 0; i < nr_parts_old; i++) {
    struct part *p = &s->parts[i];
    if (p->time_bin == time_bin_inhibited ||
        p->time_bin == time_bin_not_created || p->wind_data.decoupled)
      continue;
    if (p->wind_data.m_pending <= 0.) continue;
    const double m_p = hydro_get_mass(p);
    const double u_draw =
        random_unit_interval(p->id, e->ti_current, random_number_light_spawn);
    const long long n =
        light_spawn_count(m_p, p->wind_data.m_pending, m_l, u_draw);

    /* Specific launch energy of the reservoir; the reservoir is emptied by
     * every draw, the remainder staying in the parent */
    const double e_w = p->wind_data.E_pending / p->wind_data.m_pending;
    const int whole = light_launch_whole(m_p, p->wind_data.m_pending, m_l, u_draw);
    wind_global_stats.m_rounding +=
        p->wind_data.m_pending - (whole ? m_p : n * m_l);
    p->wind_data.m_pending = 0.;
    p->wind_data.E_pending = 0.;

    /* A parent below 2 m_l launches itself */
    if (whole) {
      const double sign = random_unit_interval(p->id, e->ti_current,
                                               random_number_wind_direction) <
                                  0.5
                              ? -1.
                              : 1.;
      const double dir[3] = {sign * props->normal[0], sign * props->normal[1],
                             sign * props->normal[2]};
      wind_light_launch(p, &s->xparts[i], props, cosmo, time, dir, e_w);
      wind_global_stats.n_spawn += 1.;
      wind_global_stats.m_spawn += m_p;
      continue;
    }
    if (n == 0) continue;

    for (long long c = 0; c < n; c++) {
      const size_t k = light_append_copy(s, i, next_id, with_gravity);
      next_id += 2;
      struct part *q = &s->parts[k];
      struct xpart *xq = &s->xparts[k];

      light_set_spawned_mass(q, xq, m_p, m_l);

      /* Offset of 0.2 h in a random direction, as particle splitting does,
       * so that no two particles coincide */
      double dx[3];
      dx[0] = random_unit_interval(q->id, e->ti_current,
                                   random_number_light_offset_1) - 0.5;
      dx[1] = random_unit_interval(q->id, e->ti_current,
                                   random_number_light_offset_2) - 0.5;
      dx[2] = random_unit_interval(q->id, e->ti_current,
                                   random_number_light_offset_3) - 0.5;
      const double norm =
          sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) + FLT_MIN;
      for (int d = 0; d < 3; d++) dx[d] *= 0.2 * q->h / norm;
      light_move(s, q, dx);

      const double sign = random_unit_interval(q->id, e->ti_current,
                                               random_number_wind_direction) <
                                  0.5
                              ? -1.
                              : 1.;
      const double dir[3] = {sign * props->normal[0], sign * props->normal[1],
                             sign * props->normal[2]};
      wind_light_launch(q, xq, props, cosmo, time, dir, e_w);

      wind_global_stats.n_spawn += 1.;
      wind_global_stats.m_spawn += m_l;
    }

    /* The parent keeps the rest */
    const double m_new = m_p - n * m_l;
    struct xpart *xp = &s->xparts[i];
    hydro_set_mass(p, m_new);
    if (p->gpart != NULL) p->gpart->mass = m_new;
    const double factor = m_p / m_new;
    chemistry_split_part(p, factor);
    cooling_split_part(p, xp, factor);
    star_formation_split_part(p, xp, factor);
    tracers_split_part(p, xp, factor);
  }

  /* Central source: copies of a template placed uniformly inside the
   * source radius and launched radially, with e_w = Edot / Mdot */
  for (long long c = 0; c < n_source; c++) {
    const size_t k = light_append_copy(s, i_template, next_id, with_gravity);
    next_id += 2;
    struct part *q = &s->parts[k];
    struct xpart *xq = &s->xparts[k];
    light_set_spawned_mass(q, xq, hydro_get_mass(&s->parts[i_template]), m_l);
    q->wind_data.m_pending = 0.;
    q->wind_data.E_pending = 0.;

    /* Uniform in the sphere: radius R u^1/3, isotropic direction */
    const double u1 =
        random_unit_interval(q->id, e->ti_current, random_number_light_offset_1);
    const double u2 =
        random_unit_interval(q->id, e->ti_current, random_number_light_offset_2);
    const double u3 =
        random_unit_interval(q->id, e->ti_current, random_number_light_offset_3);
    const double cos_t = 2. * u2 - 1.;
    const double sin_t = sqrt(fmax(0., 1. - cos_t * cos_t));
    const double phi = 2. * M_PI * u3;
    const double dir[3] = {sin_t * cos(phi), sin_t * sin(phi), cos_t};
    const double r = props->source_radius * cbrt(u1);
    for (int d = 0; d < 3; d++) {
      double x = props->source_centre[d] + r * dir[d];
      if (s->periodic) x = box_wrap(x, 0.0, s->dim[d]);
      q->x[d] = x;
      xq->v_full[d] = 0.f;
      q->v[d] = 0.f;
      if (q->gpart != NULL) {
        q->gpart->x[d] = x;
        q->gpart->v_full[d] = 0.f;
      }
    }
    wind_light_launch(q, xq, props, cosmo, time, dir,
                      props->source_Edot / props->source_Mdot);

    props->source_m_carry -= m_l;
    wind_global_stats.n_spawn += 1.;
    wind_global_stats.m_spawn += m_l;
    wind_global_stats.m_source += m_l;
  }

  if ((next_id - offset_id) != expected_count_id)
    error("Light spawn created %lld ids, expected %lld", next_id - offset_id,
          expected_count_id);

  if (e->verbose)
    message("Spawned %lld light particles (%lld from the source); took %.3f %s.",
            n_new + n_source, n_source, clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

/**
 * @brief Should this step rebuild the tree for the light particles?
 */
int engine_light_needs_rebuild(const struct engine *e) {
  const struct wind_props *props = e->wind_props;
  if (!props->light) return 0;
  return e->time - props->light_last_rebuild >= props->light_rebuild_interval;
}

#else

void engine_light_merge(struct engine *e) {}
void engine_light_spawn(struct engine *e) {}
int engine_light_needs_rebuild(const struct engine *e) { return 0; }

#endif /* WIND_TIGRESS */
