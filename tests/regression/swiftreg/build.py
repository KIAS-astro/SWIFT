"""Configure and build SWIFT out of tree, one cached build per flag set.

A build directory is named by a hash of its configure flags. The first
request configures and builds; later requests only run make, which
rebuilds what changed because automake tracks header dependencies.
The source tree is never written to.
"""

import hashlib
import json
import os
import subprocess


class BuildError(RuntimeError):
    pass


def source_dir():
    """Root of the SWIFT source tree containing this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", ".."))


def _key(flags):
    text = json.dumps(sorted(flags))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def build(flags, build_root, jobs=8, log=print):
    """Return the path of a swift binary built with the given configure flags.

    flags is the full list of configure arguments, physics and machine flags
    together. Output of configure and make goes to files in the build
    directory, and is quoted in the exception if a step fails.
    """
    src = source_dir()
    if os.path.exists(os.path.join(src, "config.status")):
        raise BuildError(
            f"{src} is configured in place, so out-of-tree builds are "
            "refused by configure. Run 'make distclean' there first."
        )

    bdir = os.path.join(build_root, _key(flags))
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, "flags.json"), "w") as f:
        json.dump(flags, f, indent=1)

    if not os.path.exists(os.path.join(bdir, "config.status")):
        log(f"  configure {os.path.basename(bdir)}: {' '.join(flags)}")
        _step([os.path.join(src, "configure")] + flags, bdir, "configure.log")

    log(f"  make {os.path.basename(bdir)}")
    _step(["make", f"-j{jobs}"], bdir, "make.log")

    binary = os.path.join(bdir, "swift")
    if not os.access(binary, os.X_OK):
        raise BuildError(f"no swift binary in {bdir}")
    return binary


def _step(cmd, cwd, logname):
    path = os.path.join(cwd, logname)
    with open(path, "w") as out:
        result = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        with open(path) as f:
            tail = f.readlines()[-40:]
        raise BuildError(
            f"{' '.join(cmd[:2])} failed in {cwd} (log {path}):\n" + "".join(tail)
        )
