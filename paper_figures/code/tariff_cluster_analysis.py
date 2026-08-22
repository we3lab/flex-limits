"""Cluster the bundled tariffs by rate structure and compare extractable value.

Supporting-information figure. Runs no optimization: it reads the `cost_optimal_cost_usd`
already solved in `processed_data/shadowcost_tariff_{mef,aef}/` and pairs it with an
inflexible reference bill, which is a cost *evaluation* of a flat 1 MW load on the same
tariff (`analysis.tariff_features.baseline_bill`, the same `eeco` call
`analysis.maxsavings.max_tariff_savings` uses for its `total_base_cost`).

The comparison is valid because `models.flexload_milp` conserves total energy
(`power_balance_rule`), so at `rte=1.0` every system config moves the same 744 MWh the flat
reference consumes. `pct_value` is therefore the net value of operating flexibly rather
than inflexibly, and it can be negative: under `uptime_equality=True` the flat profile is
infeasible, so a 25%-uptime load must run at 4x power and pays the demand charge on that
peak. That penalty against the arbitrage gain is exactly what this figure decomposes.

The reviewer-facing walkthrough of the four panels — what each axis is, which statistic backs
each claim, and the known artifacts — lives next to the render, in
`figures/SI/tariff_cluster_analysis_caption.md`.
"""

import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis import maxsavings as ms
from analysis import pricesignal as ps
from analysis.tariff_features import FEATURE_COLS, features_for_tariff
from figpaths import figure_dir

# grid parameter settings, matched to the shadow-cost sweep that produced the input data
months = [1, 7]
regions = ["SPP", "CAISO", "ERCOT", "PJM", "MISO", "NYISO", "ISONE"]

# `cost_optimal_cost_usd` is identical in the mef and aef datasets by construction — the
# cost-optimal solve never sees an emissions signal — so this figure does not depend on the
# emissions basis. `checks/audit_checks.py::mef_aef_independence` asserts it.
emissions_type = os.environ.get("EMISSIONS_TYPE", "mef")

# data/figure gen settings
generate_data = os.environ.get("GENERATE_DATA", "0") == "1"

# Silhouette does not identify a k here — it stays between 0.33 and 0.38 across k=2..8 with
# no clear optimum, which is what mixed count/continuous features with a large exact-zero
# mass tend to give. So k is chosen as the smallest value that separates the distinct rate
# archetypes. That the conclusion does not depend on the choice is reported in
# `cluster_stats_{emissions_type}.csv` rather than drawn: the spread in extractable value
# between the best and worst cluster (`cluster_median_value_spread_pp`) plateaus from k=3.
n_clusters = 4
sil_ks = list(range(2, 9))
random_state = 0

# the four system configs, in the order they are drawn, matching the `systems` dict in
# shadowcosttariff_violinplot.py so the colors line up with manuscript figure 5b
system_names = ["maxflex", "25uptime_0flex", "50uptime_50flex", "100uptime_25flex"]
system_titles = {
    "maxflex": "Max. savings",
    "25uptime_0flex": "25% up, 0% cap.",
    "50uptime_50flex": "50% up, 50% cap.",
    "100uptime_25flex": "100% up, 25% cap.",
}

# short axis labels for the clustering features
feature_titles = {
    "dem_share": "demand $ share",
    "dem_to_eng": "demand/energy $",
    "n_demand": "# demand chg.",
    "n_flat_dem": "# non-coinc. dem.",
    "n_tou_dem": "# coincident dem.",
    "dem_rate_max": "max $/kW",
    "n_energy": "# energy periods",
    "eng_spread": "energy price spread",
    "n_charges": "# charges (all)",
    "n_tiers": "# tiers",
}

with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors = colors["examplesys_colors"]

paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shadowcost_dir = os.path.join(
    paperfigs_basepath, "processed_data", f"shadowcost_tariff_{emissions_type}"
)
datadir = os.path.join(paperfigs_basepath, "processed_data", "tariff_clustering")
os.makedirs(datadir, exist_ok=True)
featurepath = os.path.join(datadir, f"tariff_features_{emissions_type}.csv")
statspath = os.path.join(datadir, f"cluster_stats_{emissions_type}.csv")


def load_shadowcost():
    """Concatenate the per-region shadow-cost results.

    `tariff_id` is a 24-hex OpenEI label and must be read as a string, or ids that happen to
    be all digits are coerced to numbers and silently fail to join.
    """
    return pd.concat(
        [
            pd.read_csv(
                os.path.join(shadowcost_dir, f"{region}.csv"),
                dtype={"tariff_id": str},
            )
            for region in regions
        ],
        ignore_index=True,
    )


def generate_features(results_df):
    """Extract tariff-structure features for every (region, month, tariff) in the results.

    One row per (region, month, tariff_id): a tariff's charges are seasonal, so its features
    differ between January and July whenever its rates are.
    """
    tariff_base = os.path.join(
        os.path.dirname(paperfigs_basepath), "data", "tariff", "bundled"
    )
    rows = []
    for month in months:
        startdate_dt, enddate_dt = ms.get_start_end(month)
        for region in regions:
            ids = results_df.loc[
                results_df["region"] == region, "tariff_id"
            ].unique()
            for tariff_id in ids:
                try:
                    tariff = pd.read_csv(
                        os.path.join(tariff_base, f"{tariff_id}.csv")
                    )
                    row = features_for_tariff(tariff, startdate_dt, enddate_dt)
                # a handful of rate sheets cannot be priced: a missing bundled file, or a
                # charge `eeco` rejects. Skipped rather than carried as NaN, since a tariff
                # with no reference bill has no value metric to cluster against.
                except Exception as e:
                    print(
                        f"SKIP {region} month{month} tariff={tariff_id}: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    continue

                row.update(region=region, month=month, tariff_id=tariff_id)
                rows.append(row)

        print(f"month {month}: {len(rows)} feature rows so far", flush=True)

    return pd.DataFrame(rows)


if generate_data:
    features_df = generate_features(load_shadowcost())
    features_df.to_csv(featurepath, index=False)
    print(
        f"wrote {len(features_df)} rows / {features_df.tariff_id.nunique()} tariffs "
        f"to {featurepath}",
        flush=True,
    )

# read in features and join them to the solved costs
features_df = pd.read_csv(featurepath, dtype={"tariff_id": str})
results_df = load_shadowcost()

# a tariff whose reference bill is zero has no meaningful percentage, and `dem_to_eng` is
# unbounded for a tariff with a demand charge and no energy charge. Clip rather than drop:
# those are real tariffs, and only the tail of the ratio is uninformative.
features_df = features_df[features_df["base_cost"] > 0].copy()
features_df["dem_to_eng"] = features_df["dem_to_eng"].fillna(
    features_df["dem_to_eng"].max()
)
features_df["dem_to_eng"] = features_df["dem_to_eng"].clip(
    upper=features_df["dem_to_eng"].quantile(0.99)
)

# cluster on the tariff structure alone, one row per (tariff, month). System uptime is a
# load-side setting, not a tariff attribute, so it enters below as a facet: that keeps the
# clusters readable as tariff archetypes and makes the uptime interaction a result.
cluster_df = features_df.drop_duplicates(subset=["tariff_id", "month"]).reset_index(
    drop=True
)
scaled = StandardScaler().fit_transform(cluster_df[FEATURE_COLS].values)

kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(scaled)
cluster_df["cluster"] = kmeans.labels_

# order clusters by how demand-charge-heavy they are, so the panels read left to right from
# energy-only tariffs to demand-charge-dominated ones and the cluster numbering is stable
# across regenerations
order = (
    cluster_df.groupby("cluster")["dem_share"].median().sort_values().index.tolist()
)
remap = {old: new for new, old in enumerate(order)}
cluster_df["cluster"] = cluster_df["cluster"].map(remap)
centroids = np.array([kmeans.cluster_centers_[old] for old in order])

# name each cluster from its own centroid rather than by hand, so a relabel after
# regeneration cannot silently contradict the legend
def cluster_label(row):
    """One-line archetype name for a cluster, read off its median feature values."""
    energy = "TOU energy" if row["eng_spread"] > 0.1 else "flat energy"
    if row["dem_share"] < 0.02:
        demand = "no demand chg."
    elif row["dem_share"] > 0.3:
        demand = "demand-dominated"
    elif row["n_tou_dem"] >= 1:
        demand = "coincident dem."
    else:
        demand = "non-coinc. dem."
    return f"{energy}, {demand}"


medians = cluster_df.groupby("cluster")[FEATURE_COLS].median()
cluster_names = {c: cluster_label(medians.loc[c]) for c in medians.index}

# join the cluster labels back onto the per-system results
merged = results_df.merge(
    cluster_df[["region", "month", "tariff_id", "cluster"]],
    on=["region", "month", "tariff_id"],
).merge(
    features_df[["region", "month", "tariff_id", "base_cost"] + FEATURE_COLS],
    on=["region", "month", "tariff_id"],
)
merged["pct_value"] = (
    100 * (merged["base_cost"] - merged["cost_optimal_cost_usd"]) / merged["base_cost"]
)
merged = merged.dropna(subset=["pct_value"])

# `maxflex` and `100uptime_25flex` can both hold the flat reference profile — the first is
# unconstrained, the second runs every hour with capacity headroom around the baseline — so
# for those two the optimum can never cost more than the reference bill and `pct_value`
# cannot legitimately be negative. Rows that are come from the 60 s solve limit and the
# relaxed-bound heuristic in `shadowcosttariff_violinplot.py`. They are kept rather than
# dropped, and the panels report medians and IQRs precisely because those are unmoved by
# them: the check below prints how far any plotted summary would shift if they were clipped.
FLAT_FEASIBLE = ["maxflex", "100uptime_25flex"]


def cluster_summary(frame):
    """Every median and quartile drawn in the value-by-cluster panel."""
    return frame.pivot_table(
        index="cluster", columns="system", values="pct_value",
        aggfunc=["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)],
    )


flat_feasible_rows = merged["system"].isin(FLAT_FEASIBLE)
n_anomalous = int((flat_feasible_rows & (merged["pct_value"] < -0.05)).sum())
clip_shift = (
    cluster_summary(
        merged.assign(
            pct_value=np.where(
                flat_feasible_rows, merged["pct_value"].clip(lower=0),
                merged["pct_value"],
            )
        )
    )
    - cluster_summary(merged)
).abs().to_numpy().max()
print(
    f"\nworse-than-baseline solves for the two configs that can hold the flat profile: "
    f"{n_anomalous} of {int(flat_feasible_rows.sum())} rows; clipping them to zero moves "
    f"no plotted median or quartile by more than {clip_shift:.3g} pp"
)

# Tariffs with no energy price variation have nothing to arbitrage regardless of their
# demand charges, so the controlled comparisons are restricted to those that do. Within
# them, the three-level split is the substantive one: a demand charge assessed on the
# monthly peak whenever it occurs cannot be scheduled around, whereas one assessed only
# during a stated time-of-use window is itself a price signal to shift against.
DC_LEVELS = ["none", "coincident only", "incl. non-coincident"]
DC_TITLES = {
    "none": "no demand\ncharge",
    "coincident only": "coincident\ndemand chg. only",
    "incl. non-coincident": "incl. non-coincident\ndemand chg.",
}


def demand_charge_level(row):
    """Which of the three demand-charge regimes a tariff-month falls into."""
    if row["dem_share"] <= 0:
        return DC_LEVELS[0]
    return DC_LEVELS[2] if row["n_flat_dem"] > 0 else DC_LEVELS[1]


tou = merged[merged["eng_spread"] > 0].copy()
tou["dc_level"] = tou.apply(demand_charge_level, axis=1)

# how much the extractable value separates across clusters, as a function of k. Reported
# alongside the silhouette so the reader can see the conclusion is not an artifact of k.
sil_scores, sep_by_k = [], []
for k in sil_ks:
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(
        scaled
    )
    sil_scores.append(silhouette_score(scaled, labels_k))
    labelled = merged.merge(
        cluster_df[["region", "month", "tariff_id"]].assign(cluster_k=labels_k),
        on=["region", "month", "tariff_id"],
    )
    spread = {}
    for system in system_names:
        med = labelled[labelled["system"] == system].groupby("cluster_k")[
            "pct_value"
        ].median()
        spread[system] = med.max() - med.min()
    sep_by_k.append(spread)

print(f"\n{len(cluster_df)} tariff-months clustered, "
      f"{cluster_df.tariff_id.nunique()} unique tariffs")
print("silhouette by k (flat, so k is chosen for interpretability): " + ", ".join(
    f"k={k}:{s:.3f}" for k, s in zip(sil_ks, sil_scores)))
print("cluster-median value spread by k, 25uptime_0flex: " + ", ".join(
    f"k={k}:{s['25uptime_0flex']:.0f}pp" for k, s in zip(sil_ks, sep_by_k)))
print("\ncluster sizes and archetypes:")
for c, n in cluster_df["cluster"].value_counts().sort_index().items():
    print(f"  C{c}: n={n:5d}  {cluster_names[c]}")
print("\nmedian pct_value by cluster x system:")
print(merged.pivot_table(
    index="cluster", columns="system", values="pct_value", aggfunc="median"
).round(1).to_string())
print("\nTOU-energy tariffs only, by demand-charge regime (median pct_value):")
print(tou.pivot_table(
    index="dc_level", columns="system", values="pct_value", aggfunc="median"
).reindex(DC_LEVELS).round(1).to_string())
print("\nMann-Whitney vs. the no-demand-charge group:")
for system in system_names:
    g = tou[tou["system"] == system]
    ref = g.loc[g["dc_level"] == DC_LEVELS[0], "pct_value"]
    parts = [f"  {system:18s} none n={len(ref):4d} med={ref.median():6.1f}%"]
    for level in DC_LEVELS[1:]:
        v = g.loc[g["dc_level"] == level, "pct_value"]
        _, p = mannwhitneyu(ref, v)
        parts.append(f"{level} n={len(v):4d} med={v.median():6.1f}% p={p:.1e}")
    print("  |  ".join(parts))


# bin edges for the dose-response on demand-charge magnitude. Fixed rather than quantile
# based so the bins mean the same thing in every panel and across regenerations, and so the
# large mass of tariffs with no demand charge at all keeps its own bin.
DEM_SHARE_BINS = [0.0, 1e-9, 0.05, 0.10, 0.20, 0.35, 1.0]


def bin_frame(frame, lo, hi):
    """The rows falling in one demand-charge-share bin.

    The first bin is exactly zero — a tariff with no demand charge at all — and the rest are
    half-open on the left, `(lo, hi]`.
    """
    if lo == 0.0:
        return frame.loc[frame["dem_share"] <= 0]
    return frame.loc[(frame["dem_share"] > lo) & (frame["dem_share"] <= hi)]


def bin_slice(frame, lo, hi):
    """The extractable values in one demand-charge-share bin."""
    return bin_frame(frame, lo, hi)["pct_value"]


def dose_response(frame, min_count=20):
    """Median extractable value in each demand-charge-share bin.

    Yields `(lo, hi, values)` per bin, skipping bins too sparse to read a median from.
    """
    for lo, hi in zip(DEM_SHARE_BINS[:-1], DEM_SHARE_BINS[1:]):
        v = bin_slice(frame, lo, hi)
        if len(v) >= min_count:
            yield lo, hi, v


def collect_stats():
    """Assemble the statistics table quoted by the SI text."""
    rows = []
    for k, s, spread in zip(sil_ks, sil_scores, sep_by_k):
        rows.append({"quantity": "silhouette", "group": f"k={k}", "value": s})
        for system in system_names:
            rows.append({
                "quantity": "cluster_median_value_spread_pp", "group": f"k={k}",
                "system": system, "value": spread[system],
            })
    for c in sorted(cluster_names):
        rows.append({
            "quantity": "cluster_size", "group": c, "label": cluster_names[c],
            "value": int((cluster_df["cluster"] == c).sum()),
        })
        for feature in FEATURE_COLS:
            rows.append({
                "quantity": "centroid_median", "group": c, "feature": feature,
                "value": medians.loc[c, feature],
            })
    for system in system_names:
        g = merged[merged["system"] == system]
        for c in sorted(cluster_names):
            v = g.loc[g["cluster"] == c, "pct_value"]
            for label, stat_value in (
                ("pct_value_median", v.median()),
                ("pct_value_mean", v.mean()),
                ("pct_value_q25", v.quantile(0.25)),
                ("pct_value_q75", v.quantile(0.75)),
                ("frac_zero_value", float((v.abs() < 0.05).mean())),
            ):
                rows.append({
                    "quantity": label, "system": system, "group": c,
                    "label": cluster_names[c], "n": len(v), "value": stat_value,
                })
        stat, p = kruskal(*[
            v["pct_value"].values for _, v in g.groupby("cluster")
        ])
        rows.append({
            "quantity": "kruskal_across_clusters", "system": system,
            "value": stat, "pvalue": p,
        })

        # the controlled comparison: among tariffs that all have arbitrage headroom, what
        # the demand-charge regime does to the value that can be realised
        t = tou[tou["system"] == system]
        ref = t.loc[t["dc_level"] == DC_LEVELS[0], "pct_value"]
        for level in DC_LEVELS:
            v = t.loc[t["dc_level"] == level, "pct_value"]
            rows.append({
                "quantity": "pct_value_median", "system": system,
                "group": f"tou_{level}", "n": len(v), "value": v.median(),
            })
            if level != DC_LEVELS[0]:
                stat, p = mannwhitneyu(ref, v)
                rows.append({
                    "quantity": "mannwhitney_vs_no_demand_charge", "system": system,
                    "group": f"tou_{level}", "n": len(ref) + len(v),
                    "value": stat, "pvalue": p,
                })

        # dose-response on demand-charge magnitude, and which structural features track the
        # value that survives once a demand charge is present
        d = t[t["dem_share"] > 0]
        for feature in FEATURE_COLS:
            rho, p = spearmanr(d[feature], d["pct_value"])
            rows.append({
                "quantity": "spearman_within_dc", "system": system, "feature": feature,
                "n": len(d), "value": rho, "pvalue": p,
            })
        for lo, hi, v in dose_response(t):
            rows.append({
                "quantity": "pct_value_median_by_dem_share", "system": system,
                "group": f"[{lo:.3f},{hi:.3f}]", "n": len(v), "value": v.median(),
            })

            # the same bin split by demand-charge regime, the panel-e classification. The
            # mix of the two shifts along the x axis, so the pooled curve is stratified here
            # to separate that composition effect from a within-group dose-response.
            for level, sub in bin_frame(t, lo, hi).groupby("dc_level"):
                rows.append({
                    "quantity": "pct_value_median_by_dem_share_and_dc_type",
                    "system": system, "group": f"[{lo:.3f},{hi:.3f}]", "label": level,
                    "n": len(sub), "value": sub["pct_value"].median(),
                })

    return pd.DataFrame(rows)


stats_df = collect_stats()
stats_df.to_csv(statspath, index=False)
print(f"\nwrote statistics to {statspath}")


# Plot the results
# define plotting defaults
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 1,
        "lines.linewidth": 1,
        "lines.markersize": 6,
        "xtick.major.size": 3,
        "xtick.major.width": 1,
        "ytick.major.size": 3,
        "ytick.major.width": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 7,
        "ytick.labelsize": 7,
        "xtick.labelsize": 7,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 150 / 25.4))
(ax_heat, ax_cluster), (ax_contrast, ax_dose) = axes


def panel_letter(ax, letter, x=-0.16):
    ax.text(x, 1.08, letter, transform=ax.transAxes, fontsize=7,
            fontweight="bold", va="top", ha="left")


def zero_line(ax):
    """Mark break-even: below it, operating flexibly costs more than not."""
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)


# Cluster identity is carried by the tick labels rather than by a color legend, so that the
# archetype a cluster stands for is readable off the axis of the panel it appears in. Built
# from `cluster_names`, which is itself read off the centroids, so a relabel after
# regeneration cannot leave a stale name behind.
cluster_counts = cluster_df["cluster"].value_counts()
cluster_tick_parts = {}
for c in sorted(cluster_names):
    energy, demand = cluster_names[c].split(", ")
    # the demand descriptor is the long one, and at four clusters across a panel the
    # neighbouring labels touch if it is left on one line
    cluster_tick_parts[c] = [f"C{c}", energy] + textwrap.wrap(demand, 14)

# pad to a common line count so the sample size sits on the same row under every cluster
tick_lines = max(len(p) for p in cluster_tick_parts.values())
cluster_ticks = [
    "\n".join(
        cluster_tick_parts[c]
        + [""] * (tick_lines - len(cluster_tick_parts[c]))
        + [f"n={cluster_counts[c]:,}"]
    )
    for c in sorted(cluster_names)
]

# (a) what distinguishes each cluster
lim = np.abs(centroids).max()
im = ax_heat.imshow(centroids.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax_heat.set(
    xticks=range(n_clusters),
    xticklabels=cluster_ticks,
    yticks=range(len(FEATURE_COLS)),
    yticklabels=[feature_titles[f] for f in FEATURE_COLS],
)
ax_heat.tick_params(axis="y", labelsize=6, length=2)
ax_heat.tick_params(axis="x", labelsize=5.5, length=0, pad=2)
ax_heat.set_title(
    f"Rate structure of each cluster ({len(cluster_df):,} tariff-months)",
    fontsize=7, pad=4,
)
cbar = fig.colorbar(im, ax=ax_heat, pad=0.03, fraction=0.045)
cbar.set_label("Centroid, SD from the mean tariff", fontsize=6, labelpad=2)
cbar.ax.tick_params(labelsize=6, length=2)
panel_letter(ax_heat, "a.", x=-0.30)

# (b) extractable value by cluster. Median with the interquartile range, rather than a box
# plot: in the clusters with no arbitrage headroom most of the distribution sits exactly at
# zero, and a box drawn there collapses to an invisible line on a symmetric log axis.
offsets = (np.arange(len(system_names)) - (len(system_names) - 1) / 2) * 0.19
for j, system in enumerate(system_names):
    g = merged[merged["system"] == system]
    x = np.arange(n_clusters) + offsets[j]
    med = np.array([g.loc[g["cluster"] == c, "pct_value"].median()
                    for c in sorted(cluster_names)])
    q25 = np.array([g.loc[g["cluster"] == c, "pct_value"].quantile(0.25)
                    for c in sorted(cluster_names)])
    q75 = np.array([g.loc[g["cluster"] == c, "pct_value"].quantile(0.75)
                    for c in sorted(cluster_names)])
    ax_cluster.vlines(x, q25, q75, color=sys_colors[system], lw=2.6, alpha=0.85)
    ax_cluster.plot(x, med, ls="none", marker="o", markersize=3.2,
                    markerfacecolor="white", markeredgecolor=sys_colors[system],
                    markeredgewidth=1.0)

zero_line(ax_cluster)
ax_cluster.set_yscale("symlog", linthresh=1, linscale=0.4)
ax_cluster.set(
    xticks=np.arange(n_clusters),
    xticklabels=cluster_ticks,
    xlim=(-0.5, n_clusters - 0.5),
    ylim=(-300, 120),
    yticks=[-100, -10, -1, 0, 1, 10, 100],
    yticklabels=["-100", "-10", "-1", "0", "1", "10", "100"],
)
ax_cluster.set_ylabel("Electricity cost savings (%)")
ax_cluster.tick_params(axis="x", labelsize=5.5, length=0, pad=2)
ax_cluster.set_title("Median and IQR by tariff cluster", fontsize=7, pad=4)
panel_letter(ax_cluster, "b.")

# (c) the controlled comparison. Every tariff here has time-varying energy rates, so all
# three groups have something to arbitrage; the only difference is the demand charge.
bar_offsets = (np.arange(len(system_names)) - (len(system_names) - 1) / 2) * 0.21
for i, level in enumerate(DC_LEVELS):
    for j, system in enumerate(system_names):
        g = tou[(tou["system"] == system) & (tou["dc_level"] == level)]
        ax_contrast.bar(i + bar_offsets[j], g["pct_value"].median(), width=0.19,
                        color=sys_colors[system], edgecolor="0.3", linewidth=0.4)

# the split is a property of the tariff, not of the load, so the group size is quoted as the
# number of distinct tariff-months. Counting rows instead would count each one up to four
# times, and unevenly: a few tariffs are missing a solve for one system config.
level_counts = {
    level: int(
        tou.loc[tou["dc_level"] == level]
        .drop_duplicates(subset=["region", "month", "tariff_id"])
        .shape[0]
    )
    for level in DC_LEVELS
}
zero_line(ax_contrast)
ax_contrast.set(
    xticks=range(len(DC_LEVELS)),
    xticklabels=[
        f"{DC_TITLES[level]}\nn={level_counts[level]:,}" for level in DC_LEVELS
    ],
    xlim=(-0.5, len(DC_LEVELS) - 0.5),
)
ax_contrast.set_ylabel("Median electricity cost savings (%)")
ax_contrast.set_title(
    "Demand-charge regime, time-varying energy rates only", fontsize=7, pad=4
)
ax_contrast.tick_params(axis="x", labelsize=6, length=0, pad=2)
panel_letter(ax_contrast, "c.")

# (d) dose-response on demand-charge magnitude, over the same tariffs as panel c. The bins
# are taken from one system config and reused for all of them, so every line is read at the
# same set of x positions even where a bin is sparse for one config.
shared_bins = [
    (lo, hi) for lo, hi, _ in dose_response(tou[tou["system"] == system_names[0]])
]
for system in system_names:
    g = tou[tou["system"] == system]
    # plot against bin index, since the bins are deliberately unequal in width
    ax_dose.plot(
        range(len(shared_bins)),
        [bin_slice(g, lo, hi).median() for lo, hi in shared_bins],
        marker="o", markersize=3, lw=1.0, color=sys_colors[system],
    )


def bin_label(lo, hi):
    """Tick label for one demand-charge-share bin, as a percentage interval."""
    if lo == 0.0:
        return "0\n(none)"
    lo_pct = 0.0 if lo < 1e-6 else lo * 100
    if hi >= 1.0:
        return f">{lo_pct:g}%"
    return f"{lo_pct:g}–{hi * 100:g}%"


zero_line(ax_dose)
ax_dose.set(
    xticks=range(len(shared_bins)),
    xticklabels=[bin_label(lo, hi) for lo, hi in shared_bins],
    xlabel="Demand charge share of the inflexible-baseline bill",
)
ax_dose.set_ylabel("Median electricity cost savings (%)")
ax_dose.set_title(
    "Demand-charge magnitude, time-varying energy rates only", fontsize=7, pad=4
)
ax_dose.tick_params(axis="x", labelsize=6)
panel_letter(ax_dose, "d.")

# shared legend for the four system configurations, which are the only thing color encodes
fig.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=sys_colors[s], edgecolor="0.3",
                      linewidth=0.4, label=system_titles[s])
        for s in system_names
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=7,
    handlelength=1.2, columnspacing=2.0,
)

fig.tight_layout(pad=0.5, w_pad=3.0, h_pad=3.0)

# save figure. Supporting information, so it is routed to figures/SI/
for figure_type in ["png", "svg", "pdf"]:
    fig.savefig(
        os.path.join(
            figure_dir(paperfigs_basepath, figure_type, si=True),
            f"tariff_cluster_analysis_{emissions_type}." + figure_type,
        ),
        dpi=300,
        bbox_inches="tight",
    )
