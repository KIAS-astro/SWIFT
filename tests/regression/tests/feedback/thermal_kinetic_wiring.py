"""The EAGLE-thermal-kinetic module, as copied from EAGLE-kinetic, behaves identically.

Test T1 of the module design (ai-notes feedback-hybrid-colibre.md). Step 1 of the
module is a copy of src/feedback/EAGLE_kinetic wired under a new configure name;
until step 2 adds the thermal channel, a run with the same initial conditions and
parameters must give the same particles bit for bit. Both runs use one thread, so
the order of floating-point operations is the same.

Set-up: the feedback budget box of feedback/_common.py with 16 stars and 10 Myr
(kicks at 600 km/s, enrichment off). Checks: positions, velocities, internal
energies and masses of all gas particles, and the star masses, identical at the
last output.
"""

import os

import numpy as np

from swiftreg.checks import Check
from swiftreg.snapshot import snapshots

from . import _common as c

DESCRIPTION = "EAGLE-thermal-kinetic before the thermal channel equals EAGLE-kinetic bit for bit"
TIER = "quick"
COMMON = ["--with-cooling=EAGLE", "--with-chemistry=EAGLE", "--with-tracers=EAGLE",
          "--with-entropy-floor=EAGLE", "--with-stars=EAGLE", "--with-star-formation=EAGLE",
          "--with-black-holes=EAGLE", "--with-ext-potential=constant",
          "--with-number-of-SNII-rays=8"]
BUILDS = {
    "kinetic": COMMON + ["--with-feedback=EAGLE-kinetic"],
    "thermal_kinetic": COMMON + ["--with-feedback=EAGLE-thermal-kinetic"],
}
FIELDS = [(0, "Coordinates"), (0, "Velocities"), (0, "InternalEnergies"), (0, "Masses"),
          (4, "Masses")]


def run(ctx):
    for name in BUILDS:
        os.makedirs(ctx.path(name))
        p, _, _ = c.setup(_Sub(ctx, ctx.path(name)), {"SNII_delta_v_km_p_s": 600.0},
                          n_stars=16, t_end=0.0102)
        ctx.swift(p, c.FLAGS, binary=name, subdir=name, name=name, threads=1)


class _Sub:
    """ctx with path() pointing into a subdirectory (setup writes ics.hdf5 there)."""

    def __init__(self, ctx, root):
        self._ctx, self._root = ctx, root
        self.data_dir, self.info, self.table = ctx.data_dir, ctx.info, ctx.table

    def path(self, *parts):
        return os.path.join(self._root, *parts)


def analyze(ctx):
    a = snapshots(ctx.path("kinetic"))[-1]
    b = snapshots(ctx.path("thermal_kinetic"))[-1]
    checks = []
    for ptype, field in FIELDS:
        x, y = a.read(ptype, field, cgs=False), b.read(ptype, field, cgs=False)
        ndiff = int(np.sum(x != y)) if x.shape == y.shape else -1
        checks.append(Check(f"PartType{ptype}/{field}: differing values", ndiff, 0,
                            kind="equal", source="step 1 is a copy of EAGLE_kinetic"))
    return checks
