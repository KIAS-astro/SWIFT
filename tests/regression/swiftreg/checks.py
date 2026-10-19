"""Checks: one measured quantity compared with one expected value."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Check:
    """A single pass/fail comparison.

    kind is one of:
      rel    |measured - expected| <= tol * |expected|
      abs    |measured - expected| <= tol
      sigma  |measured - expected| <= tol * sigma      (sigma required)
      max    measured <= expected
      min    measured >= expected
      equal  measured == expected
    source says where the expected value comes from (an equation, an exact
    identity of the scheme, a parameter), so a failure can be traced.
    """

    name: str
    measured: float
    expected: float
    kind: str = "rel"
    tol: float = 0.0
    sigma: float = None
    source: str = ""
    passed: bool = field(init=False, default=False)

    def __post_init__(self):
        m, e = float(self.measured), float(self.expected)
        if not np.isfinite(m):
            self.passed = False
        elif self.kind == "rel":
            self.passed = abs(m - e) <= self.tol * abs(e)
        elif self.kind == "abs":
            self.passed = abs(m - e) <= self.tol
        elif self.kind == "sigma":
            if self.sigma is None or self.sigma <= 0:
                raise ValueError(f"check {self.name}: sigma must be positive")
            self.passed = abs(m - e) <= self.tol * self.sigma
        elif self.kind == "max":
            self.passed = m <= e
        elif self.kind == "min":
            self.passed = m >= e
        elif self.kind == "equal":
            self.passed = m == e
        else:
            raise ValueError(f"check {self.name}: unknown kind {self.kind}")

    def describe(self):
        status = "ok  " if self.passed else "FAIL"
        if self.kind == "rel":
            err = abs(self.measured - self.expected) / abs(self.expected)
            detail = f"rel. diff {err:.3g} (tol {self.tol:g})"
        elif self.kind == "abs":
            detail = f"abs. diff {abs(self.measured - self.expected):.3g} (tol {self.tol:g})"
        elif self.kind == "sigma":
            n = abs(self.measured - self.expected) / self.sigma
            detail = f"{n:.2f} sigma (tol {self.tol:g}, sigma {self.sigma:.3g})"
        else:
            detail = self.kind
        return (
            f"{status} {self.name}: measured {self.measured:.6g}, "
            f"expected {self.expected:.6g}; {detail}"
        )

    def as_dict(self):
        return {
            "name": self.name,
            "measured": float(self.measured),
            "expected": float(self.expected),
            "kind": self.kind,
            "tol": self.tol,
            "sigma": self.sigma,
            "source": self.source,
            "passed": self.passed,
        }
