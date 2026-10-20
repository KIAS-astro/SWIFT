"""The Arkenstone I galaxy with GIZMO-MFM hydrodynamics, density recoupling with
merging into a host, and the warm channel with K20a launch speeds (long tier).

The galaxy, ISM model and diagnostics of isolated_galaxy/arkenstone_va (ICs
<data>/ics/ARK_<res>.hdf5, SWIFT_REG_ARK_RES = m2e5 by default); the hydro scheme is
SWIFT_REG_ARK_HYDRO (default gizmo-mfm, with the HLLC Riemann solver; sphenix also
works). Runs (SWIFT_REG_ARKM_RUNS, comma-separated; default all), for each of three
hot loadings L = highspec, lowspec (S24a Table 2, internal convention) and tig (K20b
hot fits at Sigma_SFR, enthalpy convention):
  L_full  hot particles recouple where the ambient n_H falls below 0.013 cm^-3 (0.1
          of the star formation threshold, as S24a's 0.1 rho_SF) or after 100 Myr and
          are merged into their nearest gas particle; plus the warm channel with
          speeds from the K20a PDF and the warm loading through TIGRESS++ mhd at
          1 kpc (10^-0.495 Sigma_SFR^-0.44), recoupling as itself at a height of
          1.5 kpc, at its apex or after 100 Myr (wind-design.md Secs. 10.3, 11.2);
  L_dh    the same without the warm channel;
  L_d2    the whole-particle wind of arkenstone_va: recoupling as itself after
          max(2 kpc, 2 kernel supports) or 25 Myr;
and nowind. Runs with host deposit split any particle above 1.5 m_g back to the base
mass at the next rebuild (arkenstone_va, density_host). Length
SWIFT_REG_ARK_TEND_MYR (default 2500), snapshots every 10 Myr. SWIFT_REG_SEED = N
builds with --with-random-seed=N.

Checks (each run):
  1. the run reached its end time;
  2. TIGRESS loadings: realised loading per phase, sum m / eta_M against M_* within
     3 sigma (arkenstone_twophase.wind_checks);
  3. host runs: the log counts merges, and every merge is a hot recoupling: the
     'merged into a host' count equals the density plus time recouplings of hot
     particles, since warm particles rejoin as themselves;
  4. L_full: launched warm energy over all launches against the K20a expectation
     within 3 sigma (arkenstone_combined, check 4);
  5. gas plus stellar mass at the end over the start: recorded, and checked to 2e-2.
     Merges and splits conserve mass exactly; the loss is outflow leaving the 2.4 Mpc
     box, which SWIFT removes (0.7 and 1.0 per cent in the SPHENIX HighSpec runs at
     1.6e6 Msun with and without merging, arkenstone-test-va.md Sec. 13).
"""

import os
import shutil

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import arkenstone_va as va
from . import arkenstone_twophase as ark2
from . import arkenstone_combined as arkc

DESCRIPTION = ("Arkenstone I galaxy, GIZMO-MFM: density recoupling with host merge "
               "plus the warm K20a channel")
TIER = "long"
HYDRO = os.environ.get("SWIFT_REG_ARK_HYDRO", "gizmo-mfm")
SEED = int(os.environ.get("SWIFT_REG_SEED", "0"))
_build = list(va.BUILDS["ark"])
if HYDRO != "sphenix":
    _build += [f"--with-hydro={HYDRO}", "--with-riemann-solver=hllc"]
if SEED:
    _build += [f"--with-random-seed={SEED}"]
BUILDS = {"ark": _build}
T_END_MYR = float(os.environ.get("SWIFT_REG_ARK_TEND_MYR", "2500"))
M_G_MSUN = {"m2e5": 2e5, "m1.6e6": 1.6e6, "m1.3e7": 1.3e7}[va.RES]

DENSITY_HOST = {"recoupling_mode": "density", "recoupling_density_H_p_cm3": 0.013,
                "recoupling_time_Myr": 100.0, "recoupling_deposit": "host"}
WARM = dict(arkc.WARM["wk20a1k"], warm_recoupling_distance_kpc=1.5,
            recoupling_overshoot_fraction=arkc.OVERSHOOT)
VARIANTS = {"full": dict(DENSITY_HOST, **WARM), "dh": dict(DENSITY_HOST), "d2": {}}
WINDS = {"nowind": {"loading_model": "none"}}
for _l, _h in arkc.HOT.items():
    for _v, _x in VARIANTS.items():
        WINDS[f"{_l}_{_v}"] = dict(_h, **_x)
_only = os.environ.get("SWIFT_REG_ARKM_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def host_params(p, m_g_msun):
    """Split a host above 1.5 m_g back to the base mass at the next rebuild."""
    if p["Wind"].get("recoupling_deposit") == "host":
        p["SPH"].update({"particle_splitting": 1,
                         "particle_splitting_mass_threshold": 1.5 * m_g_msun / 1e10})


def run(ctx):
    for t in (va.params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    ics = va.ic_file(ctx.data_dir)
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(ics, ctx.path(name, va.IC_NAME))
        p = va.parameters(ctx.data_dir, T_END_MYR, va.DT_SNAP_MYR, WINDS[name])
        host_params(p, M_G_MSUN)
        ctx.swift(p, va.FLAGS, subdir=name, name=name, timeout=48 * 3600)


def log_merges(log_path):
    """(distance, density, time, merged) from the last 'Wind recouplings by
    criterion' line of a log, or None."""
    last = None
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Wind recouplings by criterion" in line:
                last = line
    if last is None:
        return None
    import re
    return [float(a) for a in re.findall(r"(?:distance|density|time|host) ([0-9.eE+]+)",
                                         last)]


def host_checks(ctx, name, w, checks):
    """Checks 3 and 4; shared with isolated_galaxy/cgm_mfm_host."""
    log = ctx.path(name, f"{name}.log")
    if w.get("recoupling_deposit") == "host":
        c = log_merges(log)
        if c is None or len(c) < 4:
            checks.append(Check(f"{name}: recouplings line in the log", 0, 1,
                                kind="equal", source="wind_print_stats"))
        else:
            n_dist, n_dens, n_time, n_merged = c[:4]
            checks.append(Check(f"{name}: merges > 0", n_merged, 1.0, kind="min",
                                source="hot recouplings deposit into a host"))
            # Hot recouplings are by density or time and every one is merged; warm
            # recouplings are by height (counted as distance), apex or time and none
            # is merged. So density <= merged <= density + time.
            checks.append(Check(f"{name}: merged - density recouplings", n_merged - n_dens,
                                0.0, kind="min", source="every density recoupling "
                                f"merges (distance {n_dist:.0f} are warm)"))
            checks.append(Check(f"{name}: merged - density - time recouplings",
                                n_merged - n_dens - n_time, 0.0, kind="max",
                                source="warm particles rejoin as themselves"))
    if w.get("warm_launch_speed_pdf") == "K20a":
        tail = ark2.tp.phases_tail(log)
        n = int(tail["n_warm"])
        if n >= 100:
            checks.append(Check(f"{name}: warm launched energy / K20a expectation",
                                tail["E_launch_warm"] / tail["E_warm_expected"], 1.0,
                                kind="sigma", tol=3.0,
                                sigma=np.sqrt(tail["E_warm_variance"])
                                / tail["E_warm_expected"],
                                source=f"sum over all {n} warm launches (log)"))


def mass_check(name, first, last, checks):
    def total(s):
        return s.read(0, "Masses").sum() + (s.read(4, "Masses").sum()
                                            if s.has(4, "Masses") else 0.0)

    r = total(last) / total(first)
    checks.append(Check(f"{name}: gas + star mass, end / start", r, 1.0, kind="rel",
                        tol=2e-2, source="outflow leaving the box is removed"))
    return r


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        first, last = snaps[0], snaps[-1]
        checks.append(Check(f"{name}: end time [Myr]", last.time_cgs / units.MYR,
                            T_END_MYR, kind="rel", tol=1e-3,
                            source="SWIFT_REG_ARK_TEND_MYR"))
        tail = None
        if name != "nowind":
            tail = ark2.wind_checks(ctx, name, WINDS[name], last, checks)
            host_checks(ctx, name, WINDS[name], checks)
        mratio = mass_check(name, first, last, checks)
        res, _ = va.series(ctx.path(name))
        ctx.info[name] = {"wind_phases": tail, "seed": SEED, "hydro": HYDRO,
                          "mass_ratio": mratio,
                          **{k: np.asarray(v).tolist() for k, v in res.items()}}
    return checks
