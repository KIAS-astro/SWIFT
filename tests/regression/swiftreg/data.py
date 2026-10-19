"""Shared input tables, fetched once into the data directory.

A table is looked for in the data directory, then copied from any path in
the environment variable SWIFT_REGRESSION_DATA_SOURCES (colon-separated
directories that contain it, e.g. an example directory where it was
already downloaded), and only then downloaded from the SWIFT web storage.
"""

import os
import shutil
import subprocess
import tarfile

WEB = "https://virgodb.cosma.dur.ac.uk/swift-webstorage"

# name -> (URL of a tar.gz that unpacks to <name>/, file that proves presence)
TABLES = {
    "coolingtables": (f"{WEB}/CoolingTables/EAGLE/coolingtables.tar.gz",
                      "redshifts.dat"),
    "yieldtables": (f"{WEB}/YieldTables/EAGLE/yieldtables.tar.gz", None),
    "photometry": (f"{WEB}/YieldTables/EAGLE/photometry.tar.gz", None),
}


def ensure(name, data_dir, log=print):
    """Return the path of table directory `name`, fetching it if needed."""
    url, marker = TABLES[name]
    target = os.path.join(data_dir, name)
    if _present(target, marker):
        return target

    os.makedirs(data_dir, exist_ok=True)
    for src in os.environ.get("SWIFT_REGRESSION_DATA_SOURCES", "").split(":"):
        candidate = os.path.join(src, name)
        if src and _present(candidate, marker):
            log(f"  copying {name} from {src}")
            shutil.copytree(candidate, target, symlinks=False)
            return target

    log(f"  downloading {name} from {url}")
    archive = os.path.join(data_dir, f"{name}.tar.gz")
    subprocess.run(["curl", "-fsSL", "-o", archive, url], check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    os.remove(archive)
    if not _present(target, marker):
        raise RuntimeError(f"{url} did not unpack to {target}")
    return target


def _present(path, marker):
    if not os.path.isdir(path) or not os.listdir(path):
        return False
    return marker is None or os.path.exists(os.path.join(path, marker))
