# ARCHIVE — retired and stale generated data

Nothing in here should be used to build a figure. It is kept for provenance only.

Audited 2026-08-19 (two passes: `shadowcost_contour` retirement, then the figure-staleness
sweep recorded below).

## Why anything went stale

Three changes to the analysis code invalidate data generated before them. Every judgement in
this file traces back to one of them.

1. **`f586be4` reformulated the continuous-load band** ("update for time-varying baseline").
   `min_contload`/`max_contload` became per-timestep and fixed against `contload_ref[t]`, and
   the min/max penalty terms left the objective. MEF/AEF/DAM results move wherever the band
   binds. They do *not* move at the `(uptime=0, power capacity=0)` corner, where the band
   collapses — which is why the figure 3 dataset survived (see below).
2. **Two asymmetric `/1000` factors were removed** — `analysis/maxsavings.py::max_tariff_savings`
   and `analysis/emissionscost.py::shadowcost_tariff`. In both, the *flexible* load was scaled
   down by 1000 while the *baseline* was not, with `electric_consumption_units=u.MW` declared
   for both. Tariff costs computed before the fix are wrong by up to 1000x.
3. **`analysis/acc_curve.py::build_pareto_front` gained a derived MIP gap.** The CAISO-July
   wholesale front spans $2.30 of a ~$934 objective, so the inherited `1e-2` default (+/- $9.3)
   swallowed the whole front and the sweep collapsed onto the emissions-optimal point.

## Manuscript figures and where their data comes from

| Figure | Script | Data |
|---|---|---|
| 1 `designspace_plot` | `designspace_plot.py` | none — analytic schematic |
| 2 `max_savings_contours/CAISO_month7` | `maxsavings_contour.py` | `max_savings_contours/CAISO_month7.csv` |
| 3 `savings_limit_boxplot` | `maxsavings_boxplot.py` | `savings_limit_boxplot.csv`, `tariff_maxsavings_all.csv` |
| 4 `pareto_emissions_cost_curve` | `acc_curve_wholesale_vs_tariff.py` | `pareto_wholesale/`, `pareto_tariff/` |
| 5a `shadowcost_wholesale_boxplot_mef` | `shadowcostwholesale_boxplot.py` | `shadowcost_wholesale_mef/` |
| 5b `shadowcost_tariff_violinplot_aef` | `shadowcosttariff_violinplot.py` | `shadowcost_tariff_aef/` |

Supporting-information figures now render to `paper_figures/figures/SI/{png,pdf,svg}/` so that
the top level of `figures/` holds only manuscript panels. Routing lives in
`paper_figures/code/figpaths.py::figure_dir`.

## ACTIVE data — safe to build figures from

| Path (under `processed_data/`) | Generated | Evidence |
|---|---|---|
| `shadowcost_wholesale_mef/` (56 csv) | 2026-08-18 11:04–11:10 | Independence test exact. Recomputed CAISO/maxflex/month7 under current code: all five columns ratio **1.000000** |
| `shadowcost_wholesale_aef/` (56 csv) | 2026-08-18 11:10–11:16 | same run, same test |
| `shadowcost_tariff_aef/` (7 region + 56 parts) | 2026-08-18 15:40 | 7,421 valid of 7,456; region files newer than their parts |
| `shadowcost_tariff_mef/` (7 region + 56 parts) | **regenerated 2026-08-19 14:10** | replaces the pre-MW data archived here — see below |
| `savings_limit_boxplot.csv` | 2025-10-24 | **old but reproducible.** See "figure 3 dataset" below |
| `max_savings_contours/CAISO_month7.csv` | 2026-08-19 08:40 | current code; corner values agree with `savings_limit_boxplot.csv` |
| `pareto_wholesale/`, `pareto_tariff/` | regenerated 2026-08-19 20:08 | all 8 fronts carry the cost-optimal seed row |
| `timevarying_baseload_example/CAISO_month7.csv` | 2026-08-18 09:01 | matches its script |

### The MEF/AEF independence test

`cost_optimal_cost_usd` comes from a solve that never sees an emissions signal, so for a given
`(region, system, month, tariff_id)` it must be **identical** between the MEF and AEF datasets.
This is the cheapest available regression test for any dataset pair, and it is what caught the
pre-MW-units bug. After the 2026-08-19 regeneration:

```
n=7421  median=1.000000  p5=1.000000  p95=1.000000  max=1.000000   PASS
row counts match AEF exactly: SPP 1352, CAISO 840, ERCOT 392, PJM 2064,
                              MISO 1728, NYISO 576, ISONE 504
```

Before it, the same test on the live MEF region CSVs gave median **0.002** — off by ~500x.

### The figure 3 dataset is old but not stale

`savings_limit_boxplot.csv` is dated 2025-10-24 and was left unassessed by the first pass. It
reproduces exactly under current code, on all four signals:

- **MEF/AEF/DAM:** its values are identical to the `(flex=0, uptime=0)` corner of every
  `max_savings_contours` csv — 84/84 MEF, 84/84 AEF, 72/84 DAM. The 12 DAM misses were *all
  PJM*, whose contour was the stale file, not this one. That corner is where the `f586be4` band
  reformulation is a no-op, which is why a 2025 file still matches 2026 code.
- **Tariff:** CAISO July recomputed under current code gives **21.2653 %** against 21.265402 %
  stored. Under the old `/1000` the same case gives 99.92 %, so this file postdates the fix.

## ARCHIVED data — do not use

| Path | Why |
|---|---|
| `shadowcost_dam_mef_abatement1.0/` (84 csv) | Mixed code generations inside one series: CAISO months 1–11 regenerated 2026-08-17, the other six regions still 2025-09-22. Not numerically broken — regenerated CAISO vs its committed version gives a median ratio of 1.0000 — but internally inconsistent |
| `shadowcost_dam_mef_abatement0.5/` (84 csv) | Uniformly 2025-09-22; predates the current `emissionscost.py` / `flexload_milp.py` |
| `shadowcost_tariff_aef_abatement1.0_partial/` (2 csv) | Abandoned partial run, CAISO months 1 and 7 only |
| `shadowcost_tariff_mef_preMW_units/` (7 csv) | **Numerically wrong.** Pre-MW power convention; tariff costs ~1000x too low. These were still *live* until 2026-08-19 — see below |
| `pareto_noseedrow/{wholesale,tariff}/` (8 csv) | Superseded generation, missing the cost-optimal seed row — see below |
| `max_savings_contours_coarsegrid/` (24 csv) | PJM and SPP on the retired 143-point grid — see below |
| `figures/ARCHIVE/shadowcost_tariff_violinplot_mef_preMW/` | Rendered 2026-08-19 10:01 from the pre-MW region CSVs, before that run's own parts had even finished at 11:03 |
| `figures/ARCHIVE/orphans/` | Figures whose output paths no longer exist in any script — see below |
| `figures/ARCHIVE/shadowcost_tariff_violinplot.{png,svg}` | 2025-09-22 pre-suffix legacy render; not reproducible by the current script, which writes an `_{emissions_type}` suffix |

### The pre-MW MEF tariff data was archived by copy, not by move

The first pass copied `shadowcost_tariff_mef/*.csv` into `shadowcost_tariff_mef_preMW_units/`
but left the originals in place, and recorded the directory as "regenerated 2026-08-19". Only
the `parts/` had been regenerated; the concat step at `shadowcosttariff_violinplot.py:142-155`
runs only under `GENERATE_DATA=1`, so the seven region files the figure actually reads were
still the 2025-10-24 pre-MW data, md5-identical to the archived copies.

Resolved 2026-08-19 14:10 by re-running with `GENERATE_DATA=1`, which also finished the 79
tariff-cases still missing from PJM `50uptime_50flex` (month 1 was at 213/258, month 7 at
224/258). The per-tariff `parts/` resume logic meant only those 79 solves ran.

### The Pareto fronts were split across two generations

`build_pareto_front` seeds the front with the cost-optimal reference row, so a complete file has
12 rows: the seed plus 11 sweep points. Only `pareto_front_wholesale_maxflex.csv` (09:43) had
it; the other seven (09:17–09:18) had 11 rows and no seed. That is not cosmetic — the shared
`acc_curve.get_mac` takes its reference as `df["emissions"].idxmax()`, so a seedless file and a
seeded file resolve to different reference rows and different curve values. All 8 were
regenerated together.

Two rows at `abatement_fraction = 0.0` are expected, not a defect: the first is the reference
from `calc_cost_optimal`, the second is the sweep's own read on it. Where they differ, that is
the intended cost-degeneracy warning described in the `build_pareto_front` docstring.

Dead config, left in place: every `systems` dict carries a `pareto_stepsize` (0.05 / 0.1) that
`build_pareto_front` ignores — the sweep is hardcoded to `np.arange(0, 1.01, 0.1)` at
`analysis/acc_curve.py:265`.

### PJM and SPP contours were a code generation behind

24 csv (PJM and SPP, all 12 months), 2025-10-24, **143 rows** against the current **275-row**
grid. PJM's `(0,0)` corner DAM value disagreed with `savings_limit_boxplot.csv` on all 12
months (ratios 0.906 to 1.410); SPP agreed, because that corner is band-independent. Both
regenerated 2026-08-19.

The remaining 59 files (CAISO except month 7, ERCOT, ISONE, MISO, NYISO; 2026-08-17) are kept
as-is by decision, but note they carry a leading unnamed index column that the current write
path no longer emits, so the directory stays mixed-format.

### Orphan figures

Archived to `figures/ARCHIVE/orphans/`. None can be reproduced by any current script:

| File | Why |
|---|---|
| `max_savings_CAISO_month4.png` | 2025-07-16; predates the `max_savings_contours/` subdirectory convention |
| `energy_capacity_SPP_month1.svg` | 2025-09-22; the script now writes `energy_capacity.{ext}` with no region/month suffix |
| `hourly_pricesignal_CAISO_month7.{png,svg}` | the script is set to month 6; month 7 is a leftover and never had a pdf |

### Retired: the `shadowcost_wholesale_*` 25 % abatement series

32 `*_25pct_abatement.csv` files were deleted from disk; the script now writes a
`_50pct_abatement.csv` series instead. A clean generation change, not damage — the deletion is
committed rather than the files being restored here.

## Known problems NOT fixed in this pass

### Figure 3 is out of scope by decision

Two real defects were found and deliberately left alone. Anyone regenerating figure 3 must deal
with both first.

- **`tariff_maxsavings_all.csv` (2025-07-23) is zero-padded.** It has exactly 2977 rows for
  every region with no NaNs, which cannot be right — regions have between 49 and 258 tariffs.
  ERCOT is 2814 exact zeros against 588 expected tariff-months; CAISO 2059 zeros against 1260
  expected. `maxsavings_boxplot.py` boxplots each column as-is, padding included, which pins
  every tariff median at 0 and lets the `whis=(0,100)` whiskers span the padded range. It also
  holds 41 values >= 99.9 % (36 CAISO, 5 SPP) — the `/1000` signature.
- **`maxsavings_boxplot.py`'s `generate_results=True` branch cannot run.** `figpath` is
  undefined at L142; `tariff_regions.flatten()` at L131 is called on a list; L143 writes
  `savings_data` rather than `tariff_savings_data`; and the read branch at L151 expects the
  wide-format `tariff_maxsavings_all.csv` that the write branch never produces. So the file
  above cannot currently be regenerated at all.

### Definitional inconsistencies across figures

- **`maxflex` is defined three ways.** `shadowcostwholesale_boxplot.py:43` uses
  `system_uptime = 1/24` and leaves `uptime_equality` unset, so `shadowcost_wholesale` applies
  its hardcoded `uptime_equality=True` and `min_onsteps` resolves to **30** hours. Every other
  script uses `system_uptime = 0.0` with `uptime_equality = False`, giving `min_onsteps = 1`.
  Figure 5a's "maxflex" is therefore a different system from figure 4's and 5b's.
- **Tariff savings are not scale-invariant, and the two tariff paths use different load
  scales.** `ms.max_savings_point` (the contours) uses a 1 MW baseload; `maxsavings_boxplot.py`
  uses 1000 MW. Demand, fixed and tiered charges do not scale with load, so figure 2's and
  figure 3's tariff numbers are not comparable — CAISO July reads -11765 % at the contour
  corner against +21.27 % in the boxplot. MEF/AEF/DAM are unaffected, being linear in load.
- `data/tariff/metadata.csv` references ~100 tariff IDs absent from `data/tariff/bundled/`;
  `gettariff(full_list=True)` logs each as "not found" and silently drops it. Worth confirming
  that is intended before figure 5b's tariff counts go into the manuscript.

### Dead code

- **`analysis/shadowcost.py::shadow_cost` is uncalled** since `shadowcost_contour.py` was
  archived. The module was left in place as library code.
- **`analysis/acc_curve.py:332-351`** carries a stale local `systems` dict with a
  `100uptime_75flex` key that appears nowhere else in the repo.
- **`paper_figures/code/marginal_abatement_cost.py` has no `savefig` call.** The repo README
  lists it as figure 4c,d, but those panels come from `acc_curve_wholesale_vs_tariff.py`'s 2x2
  layout. It is now a diagnostic that reads the same Pareto CSVs.

## Retired analysis: `shadowcost_contour`

The `shadowcost_contour` line of analysis was inconclusive and unused, and has been retired in
full:

| What | Where it now lives |
|---|---|
| Script | `paper_figures/code/ARCHIVE/shadowcost_contour/shadowcost_contour.py` |
| Figures (~675 files, 12 dirs) | `paper_figures/figures/ARCHIVE/shadowcost_contour/{png,svg,pdf}/` |
| Data | `shadowcost_dam_mef_abatement1.0/`, `shadowcost_dam_mef_abatement0.5/`, `shadowcost_tariff_aef_abatement1.0_partial/` (this directory) |

Two notes for anyone reviving it:

- **The data paths were built from an f-string stem** (`f"shadowcost_{costing_type}_{emissions_type}"`
  plus an `_abatement<frac>` suffix), which is why a plain grep for these directory names finds
  no references. They were live inputs despite looking orphaned.
- The moved script's `os.path.dirname(os.path.dirname(__file__))` now resolves to
  `paper_figures/code/`, and its `colorscheme.json` open is relative to the new directory — it
  will not run in place without path fixes.

The archived figure directories keep their original names, which span four naming generations
(`shadowcost_contour`, `…_mef_dam`, `…_mef_dam_abatement0.5`, `…_mef_dam_abatement1.0`, plus a
single-file `…_aef_tariff` stub).
