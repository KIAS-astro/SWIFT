"""Combined wind: hot channel plus a warm channel whose launch speeds are drawn from
the K20a cool-phase outflow PDF (wind-design.md Sec. 11).

The star-forming layer of wind/launch_rate (prfm_slab, Sigma_g = 200 Msun/pc^2,
t_dep = 264 Myr). Warm speeds: v_ref ~ Gamma(2, v0), the K20a mass-flux PDF at the
reference height with v0 = 60 Sigma_SFR^0.23 + 3 km/s (K20a Eq. 24 at 1 kpc), and
v_L^2 = v_ref^2 + v_climb^2. Two runs:

  launch     zero field; hot TIGRESS K20b loadings at Sigma_SFR, warm eta_M = 1.5
             (constant), v_climb = 60.5 km/s; recoupling pushed
             beyond the run (d_rec = 1e3 kpc, t_rec = 1e4 Myr, no apex without
             gravity), 2 per cent of t_dep. Regression of the realised loadings
             per phase against the expected ones.
  ballistic  constant field g = 1e-7 cm s^-2 along -z; warm only (eta_M = 1.5,
             v_climb = 0), recoupling at the apex or at a height of
             0.6 kpc along the launch direction, decoupled time-step cap 0.01
             Myr, 8 Myr, snapshots every 0.5 Myr. Validation of the warm-particle
             trajectories against the analytic parabola.

Checks, launch:
  1. sum m / eta per phase (log) against M_* within 3 sigma (each launch stands
     for m / eta of stars; sigma^2 = (1 + 1/eta) N_* m_g^2): realised mass
     loadings = input;
  2. warm kicks (snapshot WindLaunchKicks) against the target distribution: the
     probability integral transform u = F(kick; v0(Sigma_SFR), v_climb) at each
     launch's recorded Sigma_SFR is U(0, 1) for draws from the target; one-sample
     Kolmogorov-Smirnov, D sqrt(N) < 1.949 (p = 1e-3); every kick >= v_climb;
  3. warm launched energy per mass (log, m (v_L^2/2 + gamma u)) against the mean
     of 3 v0^2 + v_climb^2/2 + gamma u over the launches within 3 standard
     errors, sd(v_L^2/2) = 21^1/2 v0^2;
  4. warm launched momentum per mass against the mean of E[v_L] (quadrature)
     within 3 standard errors, sd(v_L) from the PDF;
  5. warm energy loading E_warm / (u_* M_*) against eta_M,warm e_warm / u_*
     within 3 sigma (count noise (1 + eta)/(eta N_*) plus speed noise);
  6. hot launched energy (log) against sum over hot launches of m eta_E u_* /
     eta_M at the recorded Sigma_SFR (enthalpy convention: every hot launch
     carries eta_E u_* / eta_M; 1e-5, float Sigma_SFR); log energy and momentum
     sums equal the snapshot sums over the kicks (1e-6).
Checks, ballistic:
  7. every decoupled warm particle at every snapshot: displacement since launch
     against v tau + g tau^2 / 2 z (tau = t - t_launch), error below
     g dt tau + 1e-6 kpc: in a constant field kick-drift-kick is exact up to the
     launch kick being applied within one step dt of the recorded launch time,
     which offsets the velocity by up to g dt;
  8. warm particles recoupled below 0.6 kpc (at the apex): height against
     g tau_rec^2 / 2 within g dt_max tau_rec + 1e-6 kpc (the apex test fires in
     the first step with v . n <= 0, so v_n = g tau_rec to within g dt_max);
     their number equals the log's apex count;
  9. warm particles recoupled by height: 0.6 kpc <= h < 0.6 kpc + v_rec dt, with
     v_rec the speed at recoupling (particles launched downwards fall with the
     field and reach 0.6 kpc below their launch point, faster than their kick).
dt = min(2e-5 t_dep, 0.01 Myr) = 5.3e-3 Myr is the largest step of a decoupled
particle. Particles relaunched or turned into stars after recoupling lose their
recoupling record, so the snapshot's apex count lies between the log's count
minus (stars formed + relaunches) and the log's count.
"""

import os

import numpy as np
from scipy import integrate

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from ..star_formation import prfm_slab as slab
from . import _twophase as tp
from .two_phase_launch import BUILDS  # noqa: F401  (same build)

DESCRIPTION = "combined wind: warm speeds from the K20a PDF; loadings per phase; ballistic flight"
TIER = "quick"
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

SIGMA = 200.0  # Msun / pc^2
U_STAR = 1e51 / (95.5 * units.SOLAR_MASS)  # erg / g
U_WARM = units.u_from_T(1e4, 0.6)
GAMMA = 5.0 / 3.0
ETA_W = 1.5
V0, V_CLIMB = 60.0, 60.5  # km/s
PDF_SLOPE, PDF_FLOOR = 0.23, 3.0  # K20a Eq. 24 (Wind defaults)
G_CGS = 1e-7
D_REC_WARM = 0.6  # kpc
DT_CAP_MYR = 0.01
T_DEP_MYR = 264.03  # prfm_slab t_dep at 200 Msun/pc^2
DT_STEP_MYR = min(2e-5 * T_DEP_MYR, DT_CAP_MYR)
T_BALLISTIC_MYR = 8.0

RUNS = {
    "launch": {"loading_model": "TIGRESS", "warm_phase": 1, "warm_eta_M": ETA_W,
               "warm_launch_speed_pdf": "K20a", "warm_pdf_v0_kms": V0,
               "warm_climb_speed_kms": V_CLIMB},
    "ballistic": {"loading_model": "constant", "eta_M": 0.0, "eta_E": 0.0,
                  "warm_phase": 1, "warm_eta_M": ETA_W,
                  "warm_launch_speed_pdf": "K20a", "warm_pdf_v0_kms": V0,
                  "warm_climb_speed_kms": 0.0},
}


def scale(sigma_sfr, a=V0, b=PDF_SLOPE, c=PDF_FLOOR):
    """K20a Eq. 24 scale speed [km/s] at Sigma_SFR [Msun/kpc^2/yr], clipped as in
    src/wind_physics.h."""
    return a * np.clip(sigma_sfr, 1e-4, 1.0) ** b + c


def cdf(v, v0, vc):
    """Target CDF of the launch speed (src/wind_physics.h)."""
    x = np.sqrt(np.maximum(v * v - vc * vc, 0.0)) / v0
    return np.where(v > vc, 1.0 - (1.0 + x) * np.exp(-x), 0.0)


def ks_uniform(u):
    """One-sample Kolmogorov-Smirnov distance of u against U(0, 1)."""
    u = np.sort(u)
    n = u.size
    if n == 0:
        return 0.0
    i = np.arange(n)
    return float(max(np.max((i + 1) / n - u), np.max(u - i / n)))


def speed_moments(v0, vc):
    """E[v_L], sd(v_L) over v_ref ~ Gamma(2, v0)."""
    p = lambda x: x / v0 ** 2 * np.exp(-x / v0)  # noqa: E731
    m1 = integrate.quad(lambda x: np.sqrt(x * x + vc * vc) * p(x), 0, np.inf,
                        epsabs=0, epsrel=1e-12)[0]
    m2 = 6.0 * v0 ** 2 + vc ** 2
    return m1, np.sqrt(m2 - m1 ** 2)


def run(ctx):
    ctx.table("photometry")
    ctx.table(params.PRFM_COOLING_TABLE)
    for name, w in RUNS.items():
        os.makedirs(ctx.path(name), exist_ok=True)
        Sigma = slab.write_layer(ctx.path(name, "ics.hdf5"), SIGMA)
        t_dep = slab.t_dep_layer(Sigma) / units.UNIT_TIME
        myr = units.MYR / units.UNIT_TIME
        if name == "launch":
            t_end, dt_snap, g = 0.02 * t_dep, 0.02 * t_dep, 0.0
        else:
            t_end, dt_snap, g = T_BALLISTIC_MYR * myr, 0.5 * myr, G_CGS
        p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, dt_snap,
                              dt_max=slab.DT_MAX * t_dep, cooling=params.PRFM_COOLING)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, -g]}
        p["SPH"]["h_max"] = 2.0 * 1.2348 * slab.LX / slab.NX
        p["PRFMEntropyFloor"] = {"calibration": "OK22", "density_threshold_H_p_cm3": 0.13}
        p["PRFMStarFormation"] = {
            "calibration": "OK22", "threshold": "measured",
            "threshold_n_H_p_cm3": 0.13, "use_dark_matter": 0,
            "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
            "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
            "map_half_width_kpc": 0.5 * slab.LX}
        if name == "launch":
            rec = dict(recoupling_distance_kpc=1e3, recoupling_time_Myr=1e4,
                       warm_recoupling_distance_kpc=1e3)
        else:
            rec = dict(recoupling_distance_kpc=1e3, recoupling_time_Myr=1e4,
                       warm_recoupling_distance_kpc=D_REC_WARM,
                       recoupling_kernel_factor=0.0, max_timestep_Myr=DT_CAP_MYR)
        p["Wind"] = dict(w, launch_direction=[0.0, 0.0, 1.0], **rec)
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=3600)


def analyze_launch(ctx, checks):
    name = "launch"
    last = snapshots(ctx.path(name))[-1]
    n_star = int(last.npart[4])
    m_star = last.read(4, "InitialMasses").sum()
    m_g = np.median(last.read(0, "Masses"))
    m = last.read(0, "Masses")
    launched = last.read(0, "WindLaunchCounts", cgs=False) > 0
    phase = last.read(0, "WindPhases", cgs=False)
    hot, warm = launched & (phase == 1), launched & (phase == 2)
    eta = last.read(0, "WindLaunchMassLoadings", cgs=False)
    sig = last.read(0, "WindLaunchSigmaSFR", cgs=False)
    kick = last.read(0, "WindLaunchKicks") / 1e5  # km/s
    tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
    to_g = units.UNIT_MASS
    to_erg = units.UNIT_MASS * units.UNIT_VELOCITY ** 2
    to_p = units.UNIT_MASS * units.UNIT_VELOCITY

    for ph, sel in (("hot", hot), ("warm", warm)):
        e_med = float(np.median(eta[sel])) if sel.any() else 1.0
        checks.append(Check(f"{name}: sum m/eta ({ph}) / M_*",
                            tail[f"m_over_eta_{ph}"] * to_g / m_star, 1.0, kind="sigma",
                            tol=3.0,
                            sigma=np.sqrt((1 + 1 / e_med) * max(n_star, 1)) * m_g / m_star,
                            source=f"{int(sel.sum())} launches, N_* = {n_star}"))

    k = kick[warm]
    n = k.size
    v0 = scale(sig[warm])
    D = ks_uniform(cdf(k, v0, V_CLIMB))
    checks.append(Check(f"{name}: warm kicks KS D sqrt(N)", D * np.sqrt(n), 1.949,
                        kind="max", source=f"PIT u = F(kick; v0(Sigma_SFR), {V_CLIMB}) "
                                           f"against U(0,1); N = {n}; p = 1e-3"))
    checks.append(Check(f"{name}: min warm kick [km/s]", k.min(), V_CLIMB * (1 - 1e-6),
                        kind="min", source="v_L >= v_climb"))

    mw = m[warm]
    e_i = (3 * v0 ** 2 + 0.5 * V_CLIMB ** 2) * 1e10 + GAMMA * U_WARM  # erg/g
    E_w = tail["E_launch_warm"] * to_erg
    M_w = tail["m_warm"] * to_g
    se = np.sqrt(np.sum(mw ** 2 * 21.0 * (v0 * 1e5) ** 4)) / M_w
    e_mean = np.sum(mw * e_i) / M_w
    checks.append(Check(f"{name}: warm launched energy per mass [(km/s)^2]",
                        E_w / M_w / 1e10, e_mean / 1e10, kind="sigma", tol=3.0,
                        sigma=se / 1e10, source="3 v0^2 + v_c^2/2 + gamma u(1e4 K) at "
                                                "each launch's Sigma_SFR"))
    mom = np.array([speed_moments(x, V_CLIMB) for x in v0])
    checks.append(Check(f"{name}: warm launched momentum per mass [km/s]",
                        tail["p_launch_warm"] * to_p / M_w / 1e5,
                        np.sum(mw * mom[:, 0]) / M_w, kind="sigma", tol=3.0,
                        sigma=np.sqrt(np.sum(mw ** 2 * mom[:, 1] ** 2)) / M_w,
                        source="E[v_L] by quadrature at each launch's v0"))
    eta_E_w = ETA_W * e_mean / U_STAR
    rel_var = (1 + ETA_W) / (ETA_W * n_star) + (se / e_mean) ** 2
    checks.append(Check(f"{name}: warm eta_E realised", E_w / (U_STAR * m_star), eta_E_w,
                        kind="sigma", tol=3.0, sigma=eta_E_w * np.sqrt(rel_var),
                        source=f"eta_M,warm <e> / u_* = {eta_E_w:.4g}"))
    kw = np.sort(k)

    eta_E_hot = np.array([tp.powerlaw(tp.HOT_E_NORM["K20"], tp.HOT_E_SLOPE, x)
                          for x in sig[hot]])
    E_h_exp = np.sum(m[hot] * eta_E_hot * U_STAR / eta[hot])
    checks.append(Check(f"{name}: hot launched energy / sum m eta_E u_* / eta_M",
                        tail["E_launch_hot"] * to_erg / E_h_exp, 1.0, kind="rel", tol=1e-5,
                        source="enthalpy convention, K20b fits at the launch Sigma_SFR"))
    E_snap = np.sum(m[warm] * (0.5 * (kick[warm] * 1e5) ** 2 + GAMMA * U_WARM))
    checks.append(Check(f"{name}: warm launched energy, log / snapshot", E_w / E_snap, 1.0,
                        kind="rel", tol=1e-6, source="double sums against float kicks"))
    checks.append(Check(f"{name}: warm launched momentum, log / snapshot",
                        tail["p_launch_warm"] * to_p / np.sum(m[warm] * kick[warm] * 1e5),
                        1.0, kind="rel", tol=1e-6, source="double sums against float kicks"))
    ctx.info[name] = {"stars": n_star, "hot": int(hot.sum()), "warm": n,
                      "warm_kick_p16_50_84": np.percentile(kw, [16, 50, 84]).tolist(),
                      "KS_D_sqrtN": D * np.sqrt(n), "log": tail}


def analyze_ballistic(ctx, checks):
    name = "ballistic"
    snaps = snapshots(ctx.path(name))
    kpc = units.KPC
    g = G_CGS
    zhat = np.array([0.0, 0.0, 1.0])
    dt = DT_STEP_MYR * units.MYR
    worst, n_pts = 0.0, 0
    for s in snaps:
        if not s.has(0, "WindDecoupled"):
            continue
        dec = s.read(0, "WindDecoupled", cgs=False) == 1
        if not dec.any():
            continue
        tau = s.time_cgs - s.read(0, "WindLaunchTimes")[dec]
        v = s.read(0, "Velocities")[dec]
        dx = s.read(0, "WindDisplacements")[dec]
        pred = v * tau[:, None] + 0.5 * g * tau[:, None] ** 2 * zhat
        bound = g * dt * tau + 1e-6 * kpc
        worst = max(worst, float(np.max(np.linalg.norm(dx - pred, axis=1) / bound)))
        n_pts += int(dec.sum())
    checks.append(Check(f"{name}: max |dx - (v tau + g tau^2/2 z)| / (g dt tau)", worst,
                        1.0, kind="max",
                        source=f"{n_pts} particle-snapshots; constant field"))

    last = snaps[-1]
    t_rec = last.read(0, "WindRecouplingTimes")
    t_l = last.read(0, "WindLaunchTimes")
    rec = ((last.read(0, "WindPhases", cgs=False) == 2) & (t_rec >= t_l) & (t_rec > 0)
           & (last.read(0, "WindDecoupled", cgs=False) == 0))
    tau = t_rec[rec] - t_l[rec]
    h = last.read(0, "WindRecouplingDistances")[rec]
    v_rec = np.sqrt(2 * last.read(0, "WindRecouplingKineticEnergies")[rec])
    d_rec = D_REC_WARM * kpc
    apex = h < d_rec
    err = np.abs(h[apex] - 0.5 * g * tau[apex] ** 2) / (g * dt * tau[apex] + 1e-6 * kpc)
    checks.append(Check(f"{name}: apex height error / (g dt tau)", float(err.max()), 1.0,
                        kind="max", source=f"{int(apex.sum())} apex recouplings"))
    tail = tp.phases_tail(ctx.path(name, f"{name}.log"))
    n_lost = int(last.npart[4]) + int(
        np.sum(np.maximum(last.read(0, "WindLaunchCounts", cgs=False) - 1, 0)))
    checks.append(Check(f"{name}: apex recouplings, log - snapshot", int(tail["n_apex"])
                        - int(apex.sum()), 0, kind="min", source="log n_recouple_apex"))
    checks.append(Check(f"{name}: apex recouplings, log - snapshot - lost records",
                        int(tail["n_apex"]) - int(apex.sum()) - n_lost, 0, kind="max",
                        source=f"stars + relaunches = {n_lost}"))
    hi = ~apex
    over = (h[hi] - d_rec) / (v_rec[hi] * dt)
    checks.append(Check(f"{name}: min (h - d_rec) by height [kpc]",
                        float(((h[hi] - d_rec) / kpc).min()) if hi.any() else 0.0,
                        -1e-6, kind="min", source="h >= d_rec,warm"))
    checks.append(Check(f"{name}: max (h - d_rec) / (v_rec dt)",
                        float(over.max()) if hi.any() else 0.0, 1.0, kind="max",
                        source="recouples within one step of crossing (speed at recoupling)"))
    # Upward kicks slower than the gas's fall speed g t are past the apex at
    # launch and recouple in the launch step
    ctx.info[name] = {"recoupled": int(rec.sum()), "apex": int(apex.sum()),
                      "apex_in_launch_step": int(np.sum(tau[apex] == 0)),
                      "height": int(hi.sum()), "particle_snapshots": n_pts,
                      "max_traj_err": worst}


def analyze(ctx):
    checks = []
    analyze_launch(ctx, checks)
    analyze_ballistic(ctx, checks)
    return checks
