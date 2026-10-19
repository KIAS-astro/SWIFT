"""PRFM kernel-column surface density on uniform gas slabs of known thickness.

With PRFMStarFormation:surface_density_estimator = kernel_column the gas
surface density of a particle is Sigma_est = rho_SPH / w(0; H_k) =
rho_SPH H_k / 1.4, with H_k = 1.825742 h the support of the cubic spline
(src/prfm_physics.h, prfm_kernel_projected). For a gas slab of surface
density Sigma the midplane estimate depends only on the slab thickness in
units of H_k. A slab of N_z lattice layers spaced dz (in-plane spacing
25 pc, periodic 1 x 1 kpc) has, at a particle of the middle layer,
    rho_SPH = (Sigma / N_z) sum_j w(z_j; H_k)
when the in-plane lattice sum equals the integral (the M4 kernel's Fourier
transform vanishes at the lattice wavenumbers when H_k is a multiple of
half the spacing, and is small otherwise), so
    Sigma_est / Sigma = sum_j w(z_j; H_k) / (N_z w(0; H_k))      (layers),
and a continuous slab of the same thickness T = N_z dz has
    Sigma_est / Sigma = int_-T/2^T/2 w(z; H_k) dz / (T w(0; H_k)) (continuum),
which is H_k / (1.4 T) for T >= 2 H_k and tends to 1 for T << H_k.

Slabs of N_z = 3, 5, 7, 11, 17 layers spaced dz = 12.5 pc, and 3 layers
spaced 6.25 pc, give T / H_k = 0.4-4.8 (H_k from SWIFT's own smoothing lengths). No star formation
(threshold 1e10 cm^-3), no gravity; one output after a few steps, before any
particle moves by more than 1e-6 of a spacing.

Checks:
  1. every middle-layer particle: written PRFMGasSurfaceDensities against
     rho H_k / 1.4 from the written density and smoothing length (1e-5:
     float32 fields), the code path;
  2. median over the middle layer of Sigma_est / Sigma against the layer sum
     at the particle's own H_k (1 per cent: in-plane lattice aliasing; the
     measured spread is reported);
  3. the same against the continuum slab of thickness T (3 per cent: the
     midpoint error of replacing the continuous slab by N_z layers, which
     grows as dz / H_k; reported);
  4. the map value PRFMMapGasSurfaceDensities equals Sigma (1e-3, as
     prfm_slab);
  5. kernel_column_correction = constant (N_z = 7): written Sigma = Sigma_est
     / 0.65 (1e-5); = equilibrium (N_z = 7): written Sigma satisfies
     Sigma f(H_k / H_g) = Sigma_est with the written H_g and f of a sech^2
     layer computed here (1e-4: f from a 4000-point Simpson sum against the
     C 128-interval one, and float32 fields).
"""

import os

import numpy as np

from swiftreg import ics, params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

DESCRIPTION = "PRFM kernel-column Sigma_g on uniform slabs of thickness 0.5-4 kernel supports"
TIER = "quick"
BUILDS = {"prfm": ["--with-cooling=EAGLE", "--with-chemistry=EAGLE",
                   "--with-tracers=EAGLE", "--with-stars=EAGLE",
                   "--with-star-formation=PRFM", "--with-entropy-floor=PRFM",
                   "--with-ext-potential=constant"]}
FLAGS = ["--hydro", "--stars", "--star-formation", "--external-gravity", "--temperature"]

NX = 40
LX, LZ = 1.0, 1.0  # kpc
DZ = 0.5 * LX / NX  # kpc
LAYERS = (3, 5, 7, 11, 17)
SIGMA = 50.0  # Msun / pc^2
T0 = 1e4  # K
GAMMA_K = 1.825742
F0 = 0.65
# name: (layers, layer spacing in units of DZ, correction)
RUNS = {f"nz{n}": (n, 1.0, "none") for n in LAYERS}
RUNS.update({"nz3_dz0.5": (3, 0.5, "none"),
             "nz7_constant": (7, 1.0, "constant"),
             "nz7_equilibrium": (7, 1.0, "equilibrium")})


def w_proj(z, H):
    """Cubic-spline column through height z, per unit density [1/length]."""
    q = np.abs(np.asarray(z, dtype=float)) / H
    y = 1.0 - q
    inner = 0.0875 - 0.5 * q ** 2 + 1.5 * q ** 4 - 1.2 * q ** 5
    outer = 0.5 * y ** 4 - 0.4 * y ** 5
    return 16.0 / H * np.where(q < 0.5, inner, np.where(q < 1.0, outer, 0.0))


def slab_ratio(T, H, n=4001):
    """Continuum Sigma_est / Sigma of a uniform slab of full thickness T."""
    z = np.linspace(-0.5 * T, 0.5 * T, n)
    return np.trapezoid(w_proj(z, H), z) / (T * w_proj(0.0, H))


def layer_ratio(nz, dz, H):
    z = (np.arange(nz) - 0.5 * (nz - 1)) * dz
    return w_proj(z, H).sum() / (nz * w_proj(0.0, H))


def bias_sech2(x, n=4000):
    """f(x) = int_0^inf sech^2(u) w(u; x) du / w(0; x), Simpson."""
    u = np.linspace(0.0, min(x, 20.0), n + 1)
    g = w_proj(u, x) / np.cosh(u) ** 2
    du = u[1] - u[0]
    return du / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum()) / w_proj(0.0, x)


def write_slab(path, nz, dz):
    dx = LX / NX
    x = (np.arange(NX) + 0.5) * dx
    z = 0.5 * LZ + (np.arange(nz) - 0.5 * (nz - 1)) * dz
    xx, yy, zz = np.meshgrid(x, x, z, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    npart = len(pos)
    Sigma = SIGMA * units.SOLAR_MASS / units.PARSEC ** 2
    mass_cgs = Sigma * (LX * units.KPC) ** 2 / npart
    u = units.u_from_T(T0, units.MU_NEUTRAL) / units.UNIT_ENERGY_PER_MASS
    gas = {
        "Coordinates": pos,
        "Velocities": np.zeros((npart, 3)),
        "Masses": np.full(npart, mass_cgs / units.UNIT_MASS),
        "SmoothingLength": np.full(npart, 2.0 * dx),
        "InternalEnergy": np.full(npart, u),
        "ParticleIDs": np.arange(1, npart + 1, dtype=np.uint64),
    }
    ics.write(path, np.array([LX, LX, LZ]), gas=gas)


def run(ctx):
    ctx.table("photometry")
    ctx.table("coolingtables")
    t_end = 1e-3 * units.MYR / units.UNIT_TIME
    for name, (nz, fdz, corr) in RUNS.items():
        os.makedirs(ctx.path(name))
        write_slab(ctx.path(name, "ics.hdf5"), nz, fdz * DZ)
        p = params.eagle_base("ics.hdf5", t_end, ctx.data_dir, t_end, dt_max=0.2 * t_end)
        for sec in ("EAGLEStarFormation", "EAGLEEntropyFloor", "EAGLEFeedback"):
            p.pop(sec)
        p["ConstantPotential"] = {"g_cgs": [0.0, 0.0, 0.0]}
        p["SPH"]["h_max"] = 10.0 * LX / NX
        p["PRFMEntropyFloor"] = {"calibration": "OK22",
                                 "density_threshold_H_p_cm3": 0.13}
        p["PRFMStarFormation"] = {
            "calibration": "OK22", "threshold": "measured",
            "threshold_n_H_p_cm3": 1e10, "use_dark_matter": 0,
            "disc_centre": [0.0, 0.0, 0.0], "disc_centre_is_absolute": 0,
            "pixel_kpc": 0.1, "cylinder_radius_kpc": 0.5, "half_height_kpc": 1.0,
            "map_half_width_kpc": 0.5 * LX,
            "surface_density_estimator": "kernel_column",
            "kernel_column_correction": corr, "kernel_column_bias": F0}
        ctx.swift(p, FLAGS, subdir=name, name=name, timeout=1800)


def analyze(ctx):
    checks = []
    Sigma = SIGMA * units.SOLAR_MASS / units.PARSEC ** 2
    rows = []
    for name, (nz, fdz, corr) in RUNS.items():
        dz = fdz * DZ * units.KPC
        snap = snapshots(ctx.path(name))[-1]
        z = snap.read(0, "Coordinates")[:, 2]
        mid = np.abs(z - 0.5 * LZ * units.KPC) < 0.25 * dz
        rho = snap.read(0, "Densities")[mid]
        H = GAMMA_K * snap.read(0, "SmoothingLengths")[mid]
        Sg = snap.read(0, "PRFMGasSurfaceDensities")[mid]
        Smap = snap.read(0, "PRFMMapGasSurfaceDensities")[mid]
        est = rho / w_proj(0.0, H)
        src = f"{name}: {mid.sum()} middle-layer particles"
        if corr == "none":
            want = est
        elif corr == "constant":
            want = est / F0
        else:
            Hg = snap.read(0, "PRFMEquilibriumScaleHeights")[mid]
            f = np.array([bias_sech2(max(x, 1.0)) for x in H / Hg])
            want = est / f
        checks.append(Check(f"{name}: max |Sigma_g written / expected - 1|",
                            np.abs(Sg / want - 1).max(),
                            1e-4 if corr == "equilibrium" else 1e-5, kind="max",
                            source=src + f"; correction {corr}"))
        checks.append(Check(f"{name}: max |map Sigma_g / Sigma - 1|",
                            np.abs(Smap / Sigma - 1).max(), 1e-3, kind="max",
                            source=src + "; uniform layer seen by the map"))
        if corr != "none":
            continue
        T = nz * dz
        Hm = np.median(H)
        r_meas = est / Sigma
        r_layer = np.array([layer_ratio(nz, dz, h) for h in H])
        r_cont = np.array([slab_ratio(T, h) for h in H])
        rows.append({"run": name, "T_over_Hk": T / Hm, "Hk_pc": Hm / units.PARSEC,
                     "ratio_median": float(np.median(r_meas)),
                     "ratio_p16_p84": np.percentile(r_meas, [16, 84]).tolist(),
                     "layer_sum": float(np.median(r_layer)),
                     "continuum": float(np.median(r_cont))})
        checks.append(Check(f"{name}: median Sigma_est / layer-sum prediction",
                            np.median(r_meas / r_layer), 1.0, kind="rel", tol=0.01,
                            source=src + f", T / H_k = {T / Hm:.2f}; "
                                   "sum_j w(z_j; H_k) / (N_z w(0; H_k))"))
        checks.append(Check(f"{name}: median Sigma_est / continuum slab",
                            np.median(r_meas / r_cont), 1.0, kind="rel", tol=0.03,
                            source=src + f", T / H_k = {T / Hm:.2f}; "
                                   "int w dz / (T w(0; H_k))"))
    t = [r["T_over_Hk"] for r in rows]
    checks.append(Check("smallest T / H_k", min(t), 0.6, kind="max",
                        source="slabs span 0.5-4 kernel supports"))
    checks.append(Check("largest T / H_k", max(t), 3.5, kind="min",
                        source="slabs span 0.5-4 kernel supports"))
    ctx.info["slabs"] = rows
    return checks
