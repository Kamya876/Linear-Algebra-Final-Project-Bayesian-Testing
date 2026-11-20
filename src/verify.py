"""Shared helpers for Iteration 3 verification.

report(name, mine, ref)   -> prints max|mine-ref| and the relative error (handout format)
VerificationLog           -> collects rows for the verification table (Data, Quantity,
                             Compared with, Difference, Pass?) and writes CSV files.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGS = ROOT / "figures"


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    denom = np.linalg.norm(ref)
    rel_err = np.linalg.norm(mine - ref) / denom if denom > 0 else float("nan")
    print(f"{name:40s} max|diff| = {abs_err:.2e} rel = {rel_err:.2e}")
    return abs_err, rel_err


class VerificationLog:
    """Rows of the verification table; every number comes from a script."""

    def __init__(self, filename):
        self.filename = filename
        self.rows = []

    def add(self, data, quantity, compared, difference_text, passed):
        self.rows.append([data, quantity, compared, difference_text, "PASS" if passed else "FAIL"])
        print(f"  [{'PASS' if passed else 'FAIL'}] {data} | {quantity} vs {compared}: {difference_text}")

    def compare(self, data, quantity, compared, mine, ref, tol):
        """Max-abs-difference comparison with an absolute tolerance."""
        diff = float(np.abs(np.asarray(mine, float) - np.asarray(ref, float)).max())
        self.add(data, quantity, compared, f"max|diff| = {diff:.1e} (tol {tol:.0e})", diff <= tol)
        return diff

    def within_se(self, data, quantity, compared, mine, ref, se, k=3):
        """Difference measured in standard errors; pass if <= k."""
        d = abs(mine - ref) / se
        self.add(data, quantity, compared, f"{d:.2f} SE (limit {k})", d <= k)
        return d

    def save(self):
        DOCS.mkdir(exist_ok=True)
        path = DOCS / self.filename
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Data", "Quantity", "Compared with", "Difference", "Pass?"])
            w.writerows(self.rows)
        print(f"Saved {len(self.rows)} verification rows to {path.relative_to(ROOT)}")
        return path
