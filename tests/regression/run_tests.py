#!/usr/bin/env python3
"""Run the SWIFT physics regression tests.

Usage examples:
    ./run_tests.py --list
    ./run_tests.py                                   # all quick tests
    ./run_tests.py star_formation                    # one suite
    ./run_tests.py star_formation/sfr_law --keep     # one test, keep output
    ./run_tests.py --tier long feedback

Machine-specific configure flags (library paths, --disable-vec, CC=...) are
given with --configure-extra or the SWIFT_REGRESSION_CONFIGURE environment
variable; test modules list only the physics flags they need.

The exit status is 0 only if every selected test passes.
"""

import argparse
import importlib
import json
import os
import pkgutil
import shlex
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from swiftreg import build, data, swift  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "swift-regression")


class Context:
    """Everything a test needs; passed to run() and analyze()."""

    def __init__(self, name, run_dir, binaries, data_dir, threads, log):
        self.name = name
        self.run_dir = run_dir
        self.binaries = binaries
        self.data_dir = data_dir
        self.threads = threads
        self.log = log
        self.info = {}

    def path(self, *parts):
        return os.path.join(self.run_dir, *parts)

    def table(self, name):
        return data.ensure(name, self.data_dir, log=self.log)

    def swift(self, params, flags, binary=None, name="run", subdir=None,
              timeout=3600, threads=None):
        """Run SWIFT; binary defaults to the test's only build. threads overrides
        --threads (e.g. 1 for runs that must be bitwise reproducible)."""
        if binary is None:
            if len(self.binaries) != 1:
                raise ValueError("test has several builds; name one")
            binary = next(iter(self.binaries.values()))
        elif binary in self.binaries:
            binary = self.binaries[binary]
        rdir = self.path(subdir) if subdir else self.run_dir
        wall = swift.run(binary, params, rdir, flags,
                         threads=self.threads if threads is None else threads,
                         name=name, timeout=timeout)
        self.log(f"  ran {name}{' in ' + subdir if subdir else ''}: {wall:.1f} s")
        return rdir


def discover():
    """Return {"suite/test": module name} for every test module."""
    found = {}
    tests_pkg = os.path.join(HERE, "tests")
    for suite in sorted(os.listdir(tests_pkg)):
        sdir = os.path.join(tests_pkg, suite)
        if not os.path.isdir(sdir) or suite.startswith(("_", ".")):
            continue
        for mod in pkgutil.iter_modules([sdir]):
            if not mod.name.startswith("_"):
                found[f"{suite}/{mod.name}"] = f"tests.{suite}.{mod.name}"
    return found


def select(available, patterns, tier):
    chosen = []
    for key, modname in available.items():
        if patterns and not any(key == p or key.startswith(p.rstrip("/") + "/")
                                for p in patterns):
            continue
        module = importlib.import_module(modname)
        if tier != "all" and getattr(module, "TIER", "quick") != tier:
            continue
        chosen.append((key, module))
    unknown = [p for p in patterns
               if not any(k == p or k.startswith(p.rstrip("/") + "/")
                          for k in available)]
    if unknown:
        raise SystemExit(f"unknown tests or suites: {', '.join(unknown)}")
    return chosen


def run_one(key, module, args, extra_flags):
    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    run_dir = os.path.join(args.run_dir, key.replace("/", "__"))
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    record = {"test": key, "description": module.DESCRIPTION,
              "run_dir": run_dir, "checks": []}
    try:
        binaries = {
            name: build.build(list(flags) + extra_flags, args.build_dir,
                              jobs=args.jobs, log=log)
            for name, flags in module.BUILDS.items()
        }
        ctx = Context(key, run_dir, binaries, args.data_dir, args.threads, log)
        module.run(ctx)
        checks = module.analyze(ctx)
        if not checks:
            raise RuntimeError("analyze() returned no checks")
        for c in checks:
            log("    " + c.describe())
        record["checks"] = [c.as_dict() for c in checks]
        record["info"] = ctx.info
        record["status"] = "pass" if all(c.passed for c in checks) else "fail"
    except Exception:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
        log(record["error"])
    record["seconds"] = round(time.time() - t0, 1)

    if record["status"] == "pass" and not args.keep:
        shutil.rmtree(run_dir, ignore_errors=True)
    return record


def _to_json(obj):
    """numpy scalars and arrays in results and ctx.info."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tests", nargs="*", help="suites or suite/test to run")
    p.add_argument("--list", action="store_true", help="list tests and exit")
    p.add_argument("--tier", default="quick", choices=["quick", "long", "all"])
    p.add_argument("--threads", type=int, default=4, help="SWIFT threads per run")
    p.add_argument("--jobs", type=int, default=8, help="make -j")
    p.add_argument("--keep", action="store_true", help="keep run directories of passing tests")
    p.add_argument("--configure-extra", default=os.environ.get("SWIFT_REGRESSION_CONFIGURE", ""),
                   help="machine-specific configure flags, one string")
    p.add_argument("--build-dir", default=os.path.join(CACHE, "build"))
    p.add_argument("--run-dir", default=os.path.join(CACHE, "runs"))
    p.add_argument("--data-dir", default=os.path.join(CACHE, "data"))
    p.add_argument("--json", default=None, help="write the results to this file")
    args = p.parse_args()

    available = discover()
    if args.list:
        for key, modname in available.items():
            module = importlib.import_module(modname)
            print(f"{key:40s} [{getattr(module, 'TIER', 'quick')}] {module.DESCRIPTION}")
        return 0

    chosen = select(available, args.tests, args.tier)
    if not chosen:
        print("no tests selected")
        return 1
    extra = shlex.split(args.configure_extra)

    results = []
    for key, module in chosen:
        print(f"== {key}: {module.DESCRIPTION}", flush=True)
        results.append(run_one(key, module, args, extra))
        print(f"== {key}: {results[-1]['status'].upper()} "
              f"({results[-1]['seconds']} s)\n", flush=True)

    print("Summary")
    for r in results:
        nfail = sum(not c["passed"] for c in r["checks"])
        detail = f"{len(r['checks'])} checks, {nfail} failed" if r["checks"] else ""
        print(f"  {r['status'].upper():5s} {r['test']:40s} {r['seconds']:7.1f} s  {detail}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=_to_json)
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
