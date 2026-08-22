"""Cheap regression checks over the generated figure data.

Each check is one of the tests the 2026-08-19 staleness audit used to tell current data from
stale data, so they are worth re-running whenever a dataset is regenerated. Run from the
repository root:

    python paper_figures/code/checks/audit_checks.py
"""

import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processed_data")
BASE = os.path.normpath(BASE)
REGIONS = ["SPP", "CAISO", "ERCOT", "PJM", "MISO", "NYISO", "ISONE"]

# tariff-case counts per region, i.e. tariffs x 4 systems x 2 months
EXPECTED_TARIFF_ROWS = {
    "SPP": 1352, "CAISO": 840, "ERCOT": 392, "PJM": 2064,
    "MISO": 1728, "NYISO": 576, "ISONE": 504,
}

results = []


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {detail}")


def mef_aef_independence():
    """`cost_optimal_cost_usd` comes from a solve that never sees an emissions signal, so for a
    given (region, system, month, tariff_id) it must be identical between the MEF and AEF
    datasets. This is what caught the pre-MW-units bug: median ratio was 484x.
    """
    def load(kind):
        return pd.concat(
            [pd.read_csv(os.path.join(BASE, f"shadowcost_tariff_{kind}", f"{r}.csv"),
                         dtype={"tariff_id": str}) for r in REGIONS],
            ignore_index=True,
        )

    mef, aef = load("mef"), load("aef")
    merged = mef.merge(aef, on=["region", "system", "month", "tariff_id"],
                       suffixes=("_mef", "_aef"))
    ratio = (merged["cost_optimal_cost_usd_mef"] / merged["cost_optimal_cost_usd_aef"])
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ok = abs(ratio.median() - 1) < 1e-6 and abs(ratio.quantile(0.95) - 1) < 1e-6
    check("tariff MEF/AEF cost independence",
          ok,
          f"n={len(ratio)} median={ratio.median():.6f} "
          f"p5={ratio.quantile(0.05):.6f} p95={ratio.quantile(0.95):.6f}")

    for kind, df in (("mef", mef), ("aef", aef)):
        counts = df.groupby("region").size().to_dict()
        bad = {r: (counts.get(r), n) for r, n in EXPECTED_TARIFF_ROWS.items()
               if counts.get(r) != n}
        check(f"shadowcost_tariff_{kind} row counts",
              not bad,
              "all regions at expected counts" if not bad else f"got/expected {bad}")


def pareto_shape():
    """A complete front is the cost-optimal seed row plus 11 sweep points. Files with 11 rows
    predate the seed and resolve `get_mac`'s reference row differently.
    """
    expected = [0.0, 0.0] + [round(0.1 * i, 1) for i in range(1, 11)]
    for kind in ("wholesale", "tariff"):
        for path in sorted(glob.glob(os.path.join(BASE, f"pareto_{kind}", "*.csv"))):
            df = pd.read_csv(path)
            got = [round(v, 6) for v in df["abatement_fraction"]]
            ok = len(df) == 12 and np.allclose(got, expected, atol=1e-6)
            check(f"pareto {kind}/{os.path.basename(path)}",
                  ok,
                  f"{len(df)} rows, abatement_fraction {got[:3]}...{got[-1:]}")


def contour_corner_agreement():
    """The (flex=0, uptime=0) corner of each contour must equal the corresponding row of
    savings_limit_boxplot.csv. That corner is where the time-varying band collapses, so it is
    invariant to the `f586be4` reformulation and makes a stable cross-dataset check.
    """
    box = pd.read_csv(os.path.join(BASE, "savings_limit_boxplot.csv"))
    agree = {"mef": 0, "aef": 0, "dam": 0}
    total = 0
    misses = []
    for path in sorted(glob.glob(os.path.join(BASE, "max_savings_contours", "*.csv"))):
        region, month = os.path.basename(path)[:-4].split("_month")
        df = pd.read_csv(path)
        corner = df[(df.continuous_flex_pct == 0.0) & (df.system_uptime_pct == 0.0)]
        row = box[(box.region == region) & (box.month == int(month))]
        if corner.empty or row.empty:
            continue
        corner, row = corner.iloc[0], row.iloc[0]
        total += 1
        for sig in agree:
            if np.isclose(corner[f"max_{sig}_savings"], row[f"{sig}_savings"], rtol=1e-3):
                agree[sig] += 1
            else:
                misses.append(f"{region}_m{month}:{sig}")
    check("contour corner vs savings_limit_boxplot",
          all(v == total for v in agree.values()),
          f"of {total} region-months: " + ", ".join(f"{k} {v}" for k, v in agree.items())
          + (f"; misses {sorted(set(misses))[:8]}" if misses else ""))


def contour_grid_uniformity():
    """All contour CSVs should be on the same grid; a 143-row file is the retired coarse grid."""
    shapes = {}
    for path in sorted(glob.glob(os.path.join(BASE, "max_savings_contours", "*.csv"))):
        n = sum(1 for _ in open(path)) - 1
        shapes.setdefault(n, []).append(os.path.basename(path))
    check("max_savings_contours grid uniformity",
          len(shapes) == 1,
          "; ".join(f"{n} rows x{len(f)}" for n, f in sorted(shapes.items())))


def figure_triplets():
    """Every manuscript and SI figure should exist in all three formats."""
    figs = os.path.normpath(os.path.join(BASE, "..", "figures"))
    for label, root in (("manuscript", figs), ("SI", os.path.join(figs, "SI"))):
        stems = {}
        for ext in ("png", "pdf", "svg"):
            d = os.path.join(root, ext)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith("." + ext):
                    stems.setdefault(fn[: -len(ext) - 1], set()).add(ext)
        incomplete = {s: sorted(e) for s, e in stems.items() if len(e) != 3}
        check(f"{label} figures present in png/pdf/svg",
              not incomplete,
              f"{len(stems)} figures complete" if not incomplete else f"incomplete: {incomplete}")


if __name__ == "__main__":
    mef_aef_independence()
    pareto_shape()
    contour_corner_agreement()
    contour_grid_uniformity()
    figure_triplets()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    raise SystemExit(1 if failed else 0)
