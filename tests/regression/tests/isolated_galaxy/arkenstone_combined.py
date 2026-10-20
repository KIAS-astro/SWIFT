"""Combined decoupled wind in the Arkenstone I isolated galaxy (wind-design.md Sec. 11;
long tier): the hot channel recoupling at 10 kpc plus the warm channel, with the warm
launch speed fixed or drawn from the K20a cool-phase PDF.

The galaxy, ISM model and diagnostics of isolated_galaxy/arkenstone_va (ICs
<data>/ics/ARK_<res>.hdf5, SWIFT_REG_ARK_RES = m2e5 by default); only the wind
differs. Runs (SWIFT_REG_ARKC_RUNS, comma-separated; default all), for each of three
hot loadings L = highspec (S24a HighSpec: eta_M = 0.32, eta_E = 0.9, Mach 1,
internal convention), lowspec (S24a LowSpec: 6.41, 0.966, Mach 4) and tig (K20b hot
fits at Sigma_SFR, enthalpy convention):
  L_hot10   hot only;
  L_w100    plus the warm channel at a fixed 100 km/s (Experiment B);
  L_wk20a   plus the warm channel with speeds from the K20a PDF: v_ref ~ Gamma(2,
            v0) at 1 kpc, v0 = 60 Sigma_SFR^0.23 + 3 km/s (K20a Eq. 24), v_L^2 =
            v_ref^2 + (60.5 km/s)^2, the potential climb to 1 kpc at R = 2.5 kpc
            (wind-design.md Sec. 10.3);
  L_wv60    the same with v0 = 60 km/s at every Sigma_SFR (2-3 times the K20a
            scale; a fast-tail sensitivity run);
  L_wk20a1k L_wk20a with the warm loading through K26 mhd at 1 kpc (norm -0.495),
            the height of the PDF, instead of at 2 kpc;
and nowind. Warm loading: the K20 set, 10^-0.961 Sigma_SFR^-0.44 (K26 mhd at 2 kpc);
1e4 K. Recoupling: hot at max(10 kpc, 2 kernel supports) or 50 Myr; warm at a
height of max(1.5 kpc, 2 kernel supports) along the launch direction, 50 Myr or the
apex. t_rec = 50 Myr lets a 380 km/s TIGRESS or LowSpec particle reach 10 kpc (it
needs about 30 Myr). Length SWIFT_REG_ARK_TEND_MYR (default 2500), snapshots every
10 Myr.

SWIFT_REG_SEED = N (default 0) builds with --with-random-seed=N, which changes every
random draw (star formation, launch, direction, warm speed); seeds measure the
run-to-run scatter.

Checks (each run):
  1. the run reached its end time;
  2. TIGRESS hot and warm launches: sum of m / eta_M over the launches against the
     stellar mass formed within 3 sigma (arkenstone_twophase.wind_checks);
  3. hot particles recoupled by the last snapshot: distance >= the recoupling
     threshold (1 - 1e-6) or flight time >= 50 Myr (minus one 1 Myr step);
  3a. hot particles recoupled by distance overshoot the threshold by at most
     0.02 of it (Wind:recoupling_overshoot_fraction) plus 1e-3 for the speed
     change within a step;
  4. K20a PDF runs: launched warm energy over all launches (log) against the sum
     of the per-launch expectations, within 3 sigma of the summed variance (with
     at least 100 launches, for the normal approximation to hold). The
     kick distribution itself is tested in wind/combined_launch, not here:
     snapshot records over-represent fast kicks (slow particles fall back and
     relaunch or form stars).
"""

import os
import shutil

import numpy as np

from swiftreg import units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import arkenstone_va as va
from . import arkenstone_twophase as ark2
from ..wind import _twophase as tp

DESCRIPTION = "Arkenstone I galaxy with the combined wind: hot at 10 kpc plus warm (fixed or K20a PDF)"
TIER = "long"
SEED = int(os.environ.get("SWIFT_REG_SEED", "0"))
BUILDS = {"ark": va.BUILDS["ark"] + ([f"--with-random-seed={SEED}"] if SEED else [])}
T_END_MYR = float(os.environ.get("SWIFT_REG_ARK_TEND_MYR", "2500"))

V0, V_CLIMB = 60.0, 60.5  # km/s
# Warm loading through K26 mhd at 1 kpc (3.8 at Sigma_SFR = 3.6e-3), the height of
# the K20a PDF: log10(3.8) + 0.44 log10(3.6e-3)
WARM_NORM_1KPC = -0.495
D_REC_HOT, T_REC = 10.0, 50.0  # kpc, Myr
OVERSHOOT = 0.02  # Wind:recoupling_overshoot_fraction
GAMMA = 5.0 / 3.0
U_WARM = units.u_from_T(1e4, 0.6)

HOT = {
    "highspec": dict(va.WINDS["highspec"]),
    "lowspec": dict(va.WINDS["lowspec"]),
    "tig": {"loading_model": "TIGRESS"},
}
WARM = {
    "hot10": {},
    "w100": {"warm_phase": 1},
    "wk20a": {"warm_phase": 1, "warm_launch_speed_pdf": "K20a", "warm_pdf_v0_kms": V0,
              "warm_climb_speed_kms": V_CLIMB},
    "wv60": {"warm_phase": 1, "warm_launch_speed_pdf": "K20a", "warm_pdf_v0_kms": V0,
             "warm_pdf_slope": 0.0, "warm_pdf_floor_kms": 0.0,
             "warm_climb_speed_kms": V_CLIMB},
    "wk20a1k": {"warm_phase": 1, "warm_launch_speed_pdf": "K20a", "warm_pdf_v0_kms": V0,
                "warm_climb_speed_kms": V_CLIMB,
                "warm_eta_M_log10_norm": WARM_NORM_1KPC},
}
WINDS = {"nowind": {"loading_model": "none"}}
for _l, _h in HOT.items():
    for _w, _x in WARM.items():
        WINDS[f"{_l}_{_w}"] = dict(_h, **_x)
_only = os.environ.get("SWIFT_REG_ARKC_RUNS")
RUNS = [k for k in WINDS if not _only or k in _only.split(",")]


def recoupling():
    """Wind parameters of the combined wind beyond the loadings."""
    return dict(recoupling_distance_kpc=D_REC_HOT, recoupling_kernel_factor=2.0,
                recoupling_time_Myr=T_REC, warm_recoupling_distance_kpc=1.5,
                max_timestep_Myr=1.0, recoupling_overshoot_fraction=OVERSHOOT)


def run(ctx):
    for t in (va.params.PRFM_COOLING_TABLE, "photometry"):
        ctx.table(t)
    ics = va.ic_file(ctx.data_dir)
    for name in RUNS:
        os.makedirs(ctx.path(name), exist_ok=True)
        shutil.copy(ics, ctx.path(name, va.IC_NAME))
        p = va.parameters(ctx.data_dir, T_END_MYR, va.DT_SNAP_MYR, WINDS[name])
        p["Wind"].update(recoupling())
        ctx.swift(p, va.FLAGS, subdir=name, name=name, timeout=48 * 3600)


def combined_checks(ctx, name, w, snaps, checks, v_climb):
    last = snaps[-1]
    """Checks 3 and 4; shared with isolated_galaxy/combined_cgm."""
    if w.get("loading_model", "none") == "none":
        return
    ph = last.read(0, "WindPhases", cgs=False)
    t_r = last.read(0, "WindRecouplingTimes", cgs=False)
    t_l = last.read(0, "WindLaunchTimes", cgs=False)
    dec = last.read(0, "WindDecoupled", cgs=False)
    d = last.read(0, "WindRecouplingDistances", cgs=False)
    thr = last.read(0, "WindRecouplingDistanceThresholds", cgs=False)
    myr = units.MYR / units.UNIT_TIME
    hot = (ph == 1) & (dec == 0) & (t_r >= t_l) & (t_r > 0)
    if hot.any():
        ok = (d[hot] >= thr[hot] * (1 - 1e-6)) | ((t_r - t_l)[hot] >= (T_REC - 1.0) * myr)
        by_d = d[hot] >= thr[hot] * (1 - 1e-6)
        checks.append(Check(f"{name}: hot recouplings violating the trigger",
                            int((~ok).sum()), 0, kind="equal",
                            source=f"{int(hot.sum())} recoupled; {int(by_d.sum())} by "
                                   f"distance, median {np.median(d[hot]):.3g} kpc"))
        if by_d.any():
            over = float(np.max(d[hot][by_d] / thr[hot][by_d] - 1))
            checks.append(Check(f"{name}: max hot overshoot / d_rec", over,
                                OVERSHOOT + 1e-3, kind="max",
                                source="Wind:recoupling_overshoot_fraction; 1e-3 for the "
                                       "speed change within a step"))
    if w.get("warm_launch_speed_pdf") == "K20a":
        # Every launch enters the log's sums; snapshot records are biased towards
        # fast kicks, since slow particles fall back and relaunch or form stars
        tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
        n = int(tail["n_warm"])
        if n < 100:  # v^2 is skewed; the normal approximation needs many launches
            return
        checks.append(Check(f"{name}: warm launched energy / K20a expectation",
                            tail["E_launch_warm"] / tail["E_warm_expected"], 1.0,
                            kind="sigma", tol=3.0,
                            sigma=np.sqrt(tail["E_warm_variance"]) / tail["E_warm_expected"],
                            source=f"sum over all {n} warm launches of m (3 v0^2 + "
                                   "v_c^2/2 + gamma u) at the launch v0 (log)"))


def analyze(ctx):
    checks = []
    for name in RUNS:
        snaps = snapshots(ctx.path(name))
        last = snaps[-1]
        t_end = last.time_cgs / units.MYR
        checks.append(Check(f"{name}: end time [Myr]", t_end, T_END_MYR, kind="rel",
                            tol=1e-3, source="SWIFT_REG_ARK_TEND_MYR"))
        tail = None
        if name != "nowind":
            tail = ark2.wind_checks(ctx, name, WINDS[name], last, checks)
            combined_checks(ctx, name, WINDS[name], snaps, checks, V_CLIMB)
        res, _ = va.series(ctx.path(name))
        ctx.info[name] = {"wind_phases": tail, "seed": SEED,
                          **{k: np.asarray(v).tolist() for k, v in res.items()}}
    return checks
