"""Run a SWIFT binary on a parameter dictionary in a run directory."""

import os
import subprocess
import time

from . import params as params_mod


class RunError(RuntimeError):
    pass


def run(binary, params, run_dir, flags, threads=4, name="run", timeout=3600):
    """Write params to <name>.yml in run_dir and run SWIFT there.

    flags are the physics switches (e.g. ["--hydro", "--stars"]). The log goes
    to <name>.log. Raises RunError on a non-zero exit, a timeout, or a log
    that does not show a clean end of run. Returns the wall-clock time in s.
    """
    os.makedirs(run_dir, exist_ok=True)
    yml = os.path.join(run_dir, f"{name}.yml")
    params_mod.write(params, yml)
    logpath = os.path.join(run_dir, f"{name}.log")
    cmd = [binary, f"--threads={threads}"] + list(flags) + [os.path.basename(yml)]

    t0 = time.time()
    with open(logpath, "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RunError(f"{name}: timed out after {timeout} s (log {logpath})")
    wall = time.time() - t0

    if result.returncode != 0:
        with open(logpath, errors="replace") as f:
            tail = f.readlines()[-30:]
        raise RunError(
            f"{name}: exit status {result.returncode} (log {logpath}):\n"
            + "".join(tail)
        )
    return wall
