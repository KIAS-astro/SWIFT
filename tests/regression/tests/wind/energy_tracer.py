"""The wind energy tracer accounts for every launch and is written to snapshots.

Set-up of wind/recouple: a periodic 4 kpc box of gas (32^3 lattice,
n_H = 0.01 cm^-3, T = 1e5 K, no cooling) in a constant field g = 1e-9 cm s^-2
along z, with sixteen particles flagged WindDecoupled = 1 in the ICs, i.e.
launched at t = 0. Twelve fast ones recouple by distance after 1.9 Myr, four
slow ones by time at 3 Myr; snapshots every 0.25 Myr catch the wind before,
during and after the recouplings.

At recoupling a particle adds to its tracer its mass, its kinetic plus
internal energy m (v^2/2 + u), the potential energy gained in flight
m (phi_rec - phi_launch), and the launch energy of that flight. The global
log line carries the same sums and the launch totals.

Checks:
  1. the snapshots carry the six tracer and potential fields;
  2. at every snapshot, sum over gas of the tracer launch energy plus the
     launch energy of the particles still decoupled equals the launched
     energy of the log line, relative 1e-6 (float fields summed in double;
     the identity is exact up to that rounding);
  3. at the end, the tracer sums over gas equal the log's recoupled totals
     (kinetic + internal, potential, launch energy), relative 1e-6 of the
     launched energy, and the tracer mass equals the particle mass for each
     recoupled particle and is zero for every other;
  4. per particle, phi_rec - phi_launch = -g dz with dz the displacement in
     flight, to 1e-6 of g L (float potentials of order g L / 2, rounded to
     6e-8 relative);
  5. energy conservation in flight: sum of tracer energy plus tracer
     potential energy against the tracer launch energy, relative 1e-5 (the
     leapfrog in a constant field conserves it up to the float rounding of
     v_full, wind/recouple check 5).

A second run repeats the set-up with self-gravity (mesh 32^3, softening
0.05 kpc), where the particle's potential holds the external term as well.
Checks 1-3 as above, and:
  4'. max |dphi + g dz| / (g |dz|) < 0.05: the gas's own potential changes
      by about G rho dz^2, 3e-3 of g dz; adding the external potential a
      second time would give 1;
  5'. in-flight conservation to 1e-5 (measured 2.1e-6): the launch potential
      of a particle decoupled in the ICs is taken at its first time-step,
      with the self-gravity term; without it the error is about 3e-4.
"""

import os

import numpy as np

from swiftreg import params
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common, recouple

DESCRIPTION = "wind energy tracer: gas + in flight = launched to round-off; in snapshots"
TIER = "quick"
BUILDS = {"wind": _common.WIND_BUILD}
FLAGS = recouple.FLAGS

SG_CONSERVATION_TOL = 1e-5
FIELDS = ["WindTracerMasses", "WindTracerEnergies", "WindTracerPotentialEnergies",
          "WindTracerLaunchEnergies", "WindLaunchPotentials", "WindRecouplingPotentials"]


def run(ctx):
    ctx.table(params.PRFM_COOLING_TABLE)
    ctx.table("photometry")
    recouple.write_ics(ctx.path("ics.hdf5"))
    myr = _common.MYR
    p = _common.base_params(ctx, "ics.hdf5", recouple.T_END * myr, 0.25 * myr,
                            recouple.DT_MAX * myr, g_cgs=(0.0, 0.0, recouple.G_CGS))
    p["Wind"] = {"loading_model": "none", "recoupling_distance_kpc": recouple.D_REC,
                 "recoupling_time_Myr": recouple.T_REC}
    ctx.swift(p, FLAGS, name="tracer", timeout=1800)

    os.makedirs(ctx.path("sg"))
    recouple.write_ics(ctx.path("sg", "ics.hdf5"))
    p["Gravity"].update({"mesh_side_length": 32, "max_physical_baryon_softening": 0.05,
                         "max_physical_DM_softening": 0.05})
    ctx.swift(p, FLAGS + ["--self-gravity"], name="tracer_sg", subdir="sg", timeout=1800)


def _sums(snap):
    m = snap.read(0, "Masses", cgs=False).astype(np.float64)
    dec = snap.read(0, "WindDecoupled", cgs=False) > 0
    ek0 = snap.read(0, "WindLaunchKineticEnergies", cgs=False).astype(np.float64)
    eth0 = snap.read(0, "WindLaunchInternalEnergies", cgs=False).astype(np.float64)
    flight = np.sum(m[dec] * (ek0[dec] + eth0[dec]))
    tr = {k: snap.read(0, k, cgs=False).astype(np.float64) for k in FIELDS[:4]}
    return flight, tr, int(dec.sum())


def analyze(ctx):
    checks = _analyze(ctx, ctx.run_dir, ctx.path("tracer.log"), "", sg=False)
    checks += _analyze(ctx, ctx.path("sg"), ctx.path("sg", "tracer_sg.log"),
                       "self-gravity: ", sg=True)
    return checks


def _analyze(ctx, run_dir, log_path, tag, sg):
    checks = []
    snaps = snapshots(run_dir)
    last = snaps[-1]
    rows = _common.wind_totals(log_path)
    tail = rows[-1]
    E_launch = tail["E_kin_launch"] + tail["E_th_launch"]
    n_wind = recouple.N_FAST + recouple.N_SLOW

    missing = [k for k in FIELDS if not last.has(0, k)]
    checks.append(Check("tracer fields missing from the snapshot", len(missing), 0,
                        kind="equal", source=", ".join(FIELDS)))
    checks.append(Check("log: launches", tail["n_launch"], n_wind, kind="equal",
                        source="IC-decoupled particles count as launches at t = 0"))

    worst, n_flight = 0.0, []
    for s in snaps:
        flight, tr, nd = _sums(s)
        worst = max(worst, abs(tr["WindTracerLaunchEnergies"].sum() + flight - E_launch)
                    / E_launch)
        n_flight.append(nd)
    checks.append(Check("max over snapshots |gas tracer + in flight - launched| / launched",
                        worst, 1e-6, kind="max",
                        source="exact identity; float fields summed in double"))
    checks.append(Check("snapshots with wind both in flight and recoupled",
                        sum(0 < k < n_wind for k in n_flight), 1, kind="min",
                        source="the identity is tested mid-way, not only at the ends"))

    _, tr, nd = _sums(last)
    E_rec_log = tail["E_kin_recouple"] + tail["E_th_recouple"]
    checks.append(Check("end: |sum tracer energy - log recoupled| / launched",
                        abs(tr["WindTracerEnergies"].sum() - E_rec_log) / E_launch,
                        1e-6, kind="max", source="same sums, float against double"))
    checks.append(Check("end: |sum tracer potential - log dE_pot| / launched",
                        abs(tr["WindTracerPotentialEnergies"].sum() - tail["E_pot_recouple"])
                        / E_launch, 1e-6, kind="max", source="as above"))
    checks.append(Check("end: |sum tracer launch - log E_launch recoupled| / launched",
                        abs(tr["WindTracerLaunchEnergies"].sum() - tail["E_launch_recouple"])
                        / E_launch, 1e-6, kind="max", source="as above"))
    m = last.read(0, "Masses", cgs=False).astype(np.float64)
    rec = last.read(0, "WindRecouplingTimes", cgs=False) > 0
    tm = tr["WindTracerMasses"]
    checks.append(Check("end: max |tracer mass / m - 1| over recoupled",
                        np.abs(tm[rec] / m[rec] - 1).max(), 1e-7, kind="max",
                        source="one recoupling each"))
    checks.append(Check("end: tracer mass of the other particles", np.abs(tm[~rec]).max(),
                        0.0, kind="equal", source="never recoupled"))

    g = recouple.G_CGS / _common.ACC_UNIT
    dz = last.read(0, "WindDisplacements", cgs=False)[rec, 2].astype(np.float64)
    dphi = (last.read(0, "WindRecouplingPotentials", cgs=False)[rec].astype(np.float64)
            - last.read(0, "WindLaunchPotentials", cgs=False)[rec])
    E_tr = tr["WindTracerEnergies"].sum() + tr["WindTracerPotentialEnergies"].sum()
    E_tr0 = tr["WindTracerLaunchEnergies"].sum()
    dev = np.abs(dphi + g * dz)
    if not sg:
        checks.append(Check("max |dphi + g dz| / (g L)", (dev / (g * recouple.L)).max(),
                            1e-6, kind="max", source="phi = -g.x for the constant field; "
                                                     "float phi of order g L / 2"))
        checks.append(Check("|tracer (E + dE_pot) / tracer launch - 1|",
                            abs(E_tr / E_tr0 - 1), 1e-5, kind="max",
                            source="gravity only in flight"))
    else:
        checks.append(Check(tag + "max |dphi + g dz| / (g |dz|)", (dev / (g * np.abs(dz))).max(),
                            0.05, kind="max",
                            source="the gas's own potential changes by G rho dz^2 ~ 3e-3 g dz; "
                                   "counting the external term twice gives 1"))
        checks.append(Check(tag + "|tracer (E + dE_pot) / tracer launch - 1|",
                            abs(E_tr / E_tr0 - 1), SG_CONSERVATION_TOL, kind="max",
                            source="potential at launch taken at the first step"))
    checks = [c if c.name.startswith(tag) else
              Check(tag + c.name, c.measured, c.expected, kind=c.kind, tol=c.tol,
                    sigma=c.sigma, source=c.source) for c in checks]
    ctx.info.update({tag + k: v for k, v in {
        "E_launch": E_launch, "flight_identity_max_rel": worst,
        "decoupled_per_snapshot": n_flight,
        "tracer_E_over_launch": tr["WindTracerEnergies"].sum() / E_launch,
        "tracer_dEpot_over_launch": tr["WindTracerPotentialEnergies"].sum() / E_launch,
        "max_dphi_dev_over_g_dz": float((dev / (g * np.abs(dz))).max()),
        "flight_conservation_rel": abs(E_tr / E_tr0 - 1)}.items()})
    return checks
