"""Conservation of the kinetic SN pair kick, as a C unit test (test T8).

Builds and runs tests/testKineticPairKick (also in make check), which calls
ray_kinetic_feedback_compute_kick_velocity for 1e5 random pairs (masses within a
factor 10, gas and star velocities up to 3 times the pair kick speed) and reports
the largest relative momentum change, energy error and angle between the kick and
the pair separation. Expected: zero momentum change and exact energy (Chaikin et
al. 2023, eqs. 12-13) and kicks along r_true - r_mirr (eq. 20), to single precision.
"""

import os
import re
import subprocess

from swiftreg.checks import Check

DESCRIPTION = "kinetic pair kick conserves momentum and energy, kicks along the pair line"
TIER = "quick"
BUILDS = {"plain": []}


def run(ctx):
    bdir = os.path.dirname(ctx.binaries["plain"])
    tests_dir = os.path.join(bdir, "tests")
    with open(ctx.path("make.log"), "w") as log:
        subprocess.run(["make", "testKineticPairKick"], cwd=tests_dir, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with open(ctx.path("test.log"), "w") as log:
        result = subprocess.run([os.path.join(tests_dir, "testKineticPairKick")],
                                cwd=ctx.run_dir, stdout=log, stderr=subprocess.STDOUT)
    ctx.info["exit_status"] = result.returncode


def analyze(ctx):
    text = open(ctx.path("test.log")).read()

    def value(label):
        m = re.search(re.escape(label) + r":\s*([-+0-9.eE]+)", text)
        return float(m.group(1)) if m else float("nan")

    return [
        Check("exit status", ctx.info["exit_status"], 0, kind="equal",
              source="the C test errors out on a violation"),
        Check("max |sum m dv| / (m |dv|)", value("max |sum m dv| / (m |dv|)"), 1e-5,
              kind="max", source="Chaikin+23 eq. 13; single precision"),
        Check("max |dE / E_pair - 1|", value("max |dE / E_pair - 1|"), 1e-3, kind="max",
              source="Chaikin+23 eq. 12; single precision of v + dv"),
        Check("max angle(kick, r_true - r_mirr) [rad]",
              value("max angle(kick, r_true - r_mirr) [rad]"), 1e-3, kind="max",
              source="Chaikin+23 eq. 20"),
        Check("pairs kicked towards each other", value("pairs kicked towards each other"),
              0, kind="equal", source="the pair separates along its line"),
    ]
