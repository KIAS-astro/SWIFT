"""Radiative remnant of one SN in uniform gas with EAGLE cooling (long tier).

10^51 erg in the 32 particles nearest the centre of a 64^3 glass (SWIFT's
glassCube_64), n_H = 1 cm^-3, solar metallicity, 128 pc box (m_g = 0.26 Msun,
h = 2.5 pc); hydro and EAGLE cooling at z = 0 (Wiersma et al. 2009 tables). The
entropy floor is lowered to 10 K so that it does not hold up the cooled shell.
The ambient gas starts at 4400 K, its radiative equilibrium with these tables
(a run started at 1e4 K cooled to 4.4e3 K within 40 kyr), so P/k = 5.8e3 K cm^-3,
about twice KO15's ambient pressure. A glass replaces the lattice of the first
version (2026-10-20), whose grid imprinted 4-fold structure on the shell.

Reference: Kim & Ostriker (2015, KO15), uniform medium, n_0 = 1, solar metallicity,
values verified against the paper (ai-notes papers-part4-kim-ostriker.md):
  - numerical shell-formation time t_sf = 4.0e4 yr n^-0.59 (Eq. 22); 41.9 kyr in
    their Table 1 for n_0 = 1, defined as the time of maximum hot-gas mass;
  - momentum at t_sf, p_sf = 2.00e5 n^-0.15 Msun km/s (Eq. 25); 2.05e5 in Table 1;
  - final momentum, measured at 10 t_sf, p_final = 2.95e5 n^-0.16 Msun km/s
    (Eq. 29), reached by 4-5 t_sf and flat after.
KO15 defines hot gas with T = P/(1.1 n_H k) > 2e4 K, which is about 1e4 K true
temperature for ionised gas, the temperature of our ambient gas. This test uses a
true-temperature threshold of 5e4 K (KOR17's 1e5 K in the same convention).

Checks (tolerances: SPH against KO15's grid code, Wiersma+09 cooling with a UV
background against KO15's cooling, and an ambient pressure about twice KO15's):
  - t_sf from the maximum of M(T > 5e4 K) against 41.9 kyr, within 25 per cent
    (the 10 kyr cadence alone resolves t_sf to 12 per cent);
  - radial momentum at 41.9 kyr against 2.05e5 Msun km/s, within 15 per cent;
  - radial momentum at 419 kyr (10 t_sf) against 2.95e5 Msun km/s, within 20 per
    cent. The first run (2026-10-20, to 300 kyr) found 3.7e5 and still rising,
    25 per cent above Eq. 29; the cause is not established (see the experiments
    note, ai-notes sn-experiments.md).
"""

import os

import numpy as np

from swiftreg import params, units
from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "single SN with EAGLE cooling: shell formation and momentum against KO15"
TIER = "long"
# Hydro scheme from SWIFT_REG_HYDRO (default SPHENIX), to compare how schemes treat the
# hot interior (mixing across the contact discontinuity).
HYDRO = os.environ.get("SWIFT_REG_HYDRO", "sphenix")
BUILDS = {"eagle": ["--with-subgrid=EAGLE"] + ([] if HYDRO == "sphenix" else
                                                [f"--with-hydro={HYDRO}"]
                                                + (["--with-riemann-solver=hllc"]
                                                   if HYDRO.startswith("gizmo") else []))}

N = 64
BOX = 0.128  # kpc
NH = 1.0
T = 4.4e3
DT_SNAP = 1.0226e-5  # internal units, 10.0 kyr
T_END = 4.6e-4  # 450 kyr
T_HOT = 5e4

KO15_T_SF_YR = 4.19e4  # Table 1, n_0 = 1
KO15_P_SF = 2.05e5  # Msun km/s, Table 1
KO15_P_FINAL = 2.95e5  # Msun km/s, Eq. 29 at n_0 = 1


def run(ctx):
    ctx.table("coolingtables")
    from swiftreg import data
    glass = data.glass(64, ctx.data_dir, log=ctx.log)
    _, rho0 = c.blast_ics(ctx.path("ics.hdf5"), N, BOX, NH, T, glass_file=glass)
    ctx.info["rho0"] = rho0
    p = params.eagle_base("ics.hdf5", T_END, ctx.data_dir, DT_SNAP, dt_max=1e-6)
    p["Snapshots"]["compression"] = 4
    p["EAGLEEntropyFloor"].update({"Jeans_temperature_norm_K": 10.0,
                                   "Cool_temperature_norm_K": 10.0})
    if os.environ.get("SWIFT_REG_SNR_OUTPUTS") == "dense":
        # 150 outputs log-spaced from 1 kyr to the end, for movies
        times = np.logspace(np.log10(1.0226e-6), np.log10(T_END), 150)
        np.savetxt(ctx.path("output_list.txt"), times, header="Time", comments="# ")
        p["Snapshots"].update({"output_list_on": 1, "output_list": "output_list.txt"})
    ctx.swift(p, ["--hydro", "--cooling", "--limiter"], timeout=4 * 3600)


def history(run_dir, box_kpc=BOX, t_hot=T_HOT):
    """Rows of t [yr], M(T > t_hot) [Msun], p_r [Msun km/s], E_total [erg]."""
    box_cm = box_kpc * units.UNIT_LENGTH
    rows = []
    for s in snapshots(run_dir):
        d = c.radial(s, box_cm)
        hot = np.sum(d["m"][d["T"] > t_hot]) / units.SOLAR_MASS
        p_r = np.sum(d["m"] * d["vr"]) / (units.SOLAR_MASS * 1e5)
        e = np.sum(d["m"] * d["u"]) + 0.5 * np.sum(d["m"] * np.sum(d["v"] ** 2, axis=1))
        rows.append([s.time_cgs / units.YEAR, hot, p_r, e])
    return np.array(rows)


def analyze(ctx):
    rows = history(ctx.run_dir)
    ctx.info["t_yr_Mhot_Msun_pr_Msunkms_E_erg"] = rows.tolist()
    t, p_r = rows[:, 0], rows[:, 2]
    t_sf = t[int(np.argmax(rows[:, 1]))]
    return [
        Check("shell formation time, max M(T > 5e4 K) [yr]", t_sf, KO15_T_SF_YR,
              kind="rel", tol=0.25, source="KO15 Table 1 (Eq. 22), n_0 = 1"),
        Check("radial momentum at 41.9 kyr [Msun km/s]", np.interp(KO15_T_SF_YR, t, p_r),
              KO15_P_SF, kind="rel", tol=0.15, source="KO15 Table 1 (Eq. 25), n_0 = 1"),
        Check("radial momentum at 10 t_sf = 419 kyr [Msun km/s]",
              np.interp(10 * KO15_T_SF_YR, t, p_r), KO15_P_FINAL, kind="rel", tol=0.20,
              source="KO15 Eq. 29, n_0 = 1"),
    ]
