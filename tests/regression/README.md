# SWIFT physics regression tests

Each test runs SWIFT on a small problem with a known answer and compares the
output with that answer: an equation from the paper that defines the model,
or an exact identity of the scheme. Each check states where its expected value
comes from and why its tolerance is what it is.

## Running

```
export SWIFT_REGRESSION_CONFIGURE="--with-hdf5=... --with-gsl=... --disable-vec CC=mpicc"
./run_tests.py --list
./run_tests.py                      # all quick tests
./run_tests.py star_formation       # one suite
./run_tests.py feedback/energy_budget_thermal --keep
```

`SWIFT_REGRESSION_CONFIGURE` (or `--configure-extra`) carries the
machine-specific configure flags; test modules list physics flags only.
The exit status is 0 only if every selected test passes; `--json FILE`
writes the results.

The source tree must not be configured in place (`make distclean`): each
set of configure flags is built out of tree in
`~/.cache/swift-regression/build/<hash>/`, configured once and afterwards
only re-made. Run directories go to `~/.cache/swift-regression/runs/` and
are deleted when a test passes unless `--keep` is given. Input tables are
fetched once into `~/.cache/swift-regression/data/`; set
`SWIFT_REGRESSION_DATA_SOURCES` to directories that already hold them to
copy instead of download.

Requires Python 3 with numpy and h5py.

## Writing a test

A module `tests/<suite>/<name>.py` defines

```python
DESCRIPTION = "one sentence"
TIER = "quick"                    # or "long"
BUILDS = {"name": [configure flags]}
def run(ctx): ...                 # write ICs and parameters, call ctx.swift()
def analyze(ctx): ...             # return a list of swiftreg.checks.Check
```

`tests/framework/uniform_box.py` is the smallest complete example.
