# imports
import os, json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from analysis.pricesignal import getmef, getdam, getaef, gettariff
from analysis.maxsavings import (
    max_mef_savings,
    max_aef_savings,
    max_dam_savings,
    max_tariff_savings,
    max_savings_point,
    get_start_end
)
GENERATE_DATA = True
N_JOBS = 40
# import color maps as json
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)

cmaps=colors["incentive_cmaps"]
sys_colors=colors["examplesys_colors"]

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

UPTIMES = np.arange(0, 25, 1) / 24  # 1 to 24 hours to percent in intervals
CONTINUOUS_FLEX = np.arange(0, 1.01, 0.1)  # 0 to 100%


def generate_data(region, months):
    """
    Generate the maximum savings contours for a region across `months`.

    Every (month, uptime, power capacity) point goes into a single fan-out. Solving one
    month at a time instead leaves most workers idle through the tail of each month,
    where a handful of hard month-long tariff MILPs are still running.
    """
    # workers import `max_savings_point` and load their own signals, so each task carries
    # only four scalars. Gurobi is held to one thread per worker: the parallelism is
    # across grid points, and letting each worker grab the machine oversubscribes it.
    # batch_size=1 because runtimes are wildly uneven: a handful of month-long tariff
    # MILPs take minutes while most points take seconds, and the default batching would
    # park a slow point at the back of a worker's queue behind other work.
    grid = [(m, u, f) for m in months for u in UPTIMES for f in CONTINUOUS_FLEX]
    results = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=1)(
        delayed(max_savings_point)(region, m, u, f, 1) for m, u, f in grid
    )

    figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    per_month = {}
    stride = len(UPTIMES) * len(CONTINUOUS_FLEX)

    for i, month in enumerate(months):
        # row-major over uptimes, matching the grid order above
        mef, aef, dam, tariff = (
            np.array(x) for x in zip(*results[i * stride:(i + 1) * stride])
        )
        df = pd.DataFrame(
            {
                "continuous_flex_pct": np.tile(CONTINUOUS_FLEX, len(UPTIMES)) * 100,
                "system_uptime_pct": np.repeat(UPTIMES, len(CONTINUOUS_FLEX)) * 100,
                "max_mef_savings": mef,
                "max_aef_savings": aef,
                "max_dam_savings": dam,
                "max_tariff_savings": tariff,
            }
        )
        # save the dataframe to a csv file
        df.to_csv(
            os.path.join(figpath, "processed_data", "max_savings_contours", f"{region}_month{month}.csv"),
            index=False,
        )
        per_month[month] = df

    return per_month

def generate_figure(df, region, month, overlay_points=[], overlay=True, save=False):
    """
    """    
    # get the region and month from the dataframe
    uptimes = df["system_uptime_pct"].unique() / 100
    continuous_flex = df["continuous_flex_pct"].unique() / 100
    # use pivot to get the max savings results
    max_mef_savings_results = df.pivot(
        index="system_uptime_pct",
        columns="continuous_flex_pct",
        values="max_mef_savings"
    )
    max_aef_savings_results = df.pivot(
        index="system_uptime_pct",
        columns="continuous_flex_pct",
        values="max_aef_savings"
    )
    max_dam_savings_results = df.pivot(
        index="system_uptime_pct",
        columns="continuous_flex_pct",
        values="max_dam_savings"
    )
    max_tariff_savings_results = df.pivot(
        index="system_uptime_pct",
        columns="continuous_flex_pct",
        values="max_tariff_savings"
    )

    # create figure - 2x2 grid for MEF[0,0], AEF[1,0], DAM[0,1], Tariffs[1,1] savings
    fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 120 / 25.4), layout="tight")

    # plot max MEF savings
    contour = ax[0, 0].contourf(
        continuous_flex * 100,
        uptimes * 100,
        max_mef_savings_results,
        levels=np.arange(0,30.1, 2.5),
        extend="max",
        cmap=cmaps["mef"],
    )
    cbar = fig.colorbar(contour, ax=ax[0, 0])

    # ax[0, 0].set_title("Marginal Emissions")
    cbar.set_label("Marginal emissions savings (%)")

    # plot max AEF savings
    contour = ax[1,0].contourf(
        continuous_flex * 100, 
        uptimes * 100, 
        max_aef_savings_results, 
        levels=np.arange(0,30.1, 2.5),
        vmax=30,
        extend="max",
        cmap=cmaps["aef"],
    )
    cbar = fig.colorbar(contour, ax=ax[1,0])
    # ax[1, 0].set_title("Average Emissions")
    cbar.set_label("Average emissions savings (%)")

    # plot max DAM savings
    contour = ax[0, 1].contourf(
        continuous_flex * 100,
        uptimes * 100,
        max_dam_savings_results,
        extend="max",
        levels=np.arange(0,50.1, 5),
        cmap=cmaps["dam"],
    )
    cbar = fig.colorbar(contour, ax=ax[0, 1])
    # ax[0, 1].set_title("Day-ahead Prices")
    cbar.set_label("Day-ahead market savings (%)")

    # plot max Tariff savings
    contour = ax[1,1].contourf(
        continuous_flex * 100, 
        uptimes * 100, 
        max_tariff_savings_results, 
        extend="max", 
        levels=np.arange(0, 51, 5),
        cmap=cmaps["tariff"],
    )
    cbar = fig.colorbar(contour, ax=ax[1,1])
    # ax[1, 1].set_title(f"Sample Tariff: {region}")
    cbar.set_label("Tariff savings (%)")
    
    # Define labels for your subplots
    subplot_labels = ['a.', 'b.', 'c.', 'd.']

    for idx, a in enumerate(ax.flatten()):
        a.set_aspect("equal", adjustable="box")
        a.set(
            xlim=(0, 100),
            xticks=np.arange(0, 101, 10),
            yticks=np.arange(0, 101, 10),
            ylim=(2, 100),
            aspect="equal",
        )
        a.set_xlabel("Power Capacity (%)", labelpad=1)
        a.set_ylabel("System Uptime (%)", labelpad=1)

        a.text(-0.2, 1.04, subplot_labels[idx], transform=a.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left')

        if overlay:
            overlay_colors= [sys_colors["maxflex"], 
                            sys_colors["25uptime_0flex"],
                            sys_colors["50uptime_50flex"],
                            sys_colors["100uptime_25flex"]]

            if idx == 3:
                # remove the 1st overlay pair and replace it with the index of the max 
                # from the df find the tariff maximizing continuous flex and uptime
                df_max = df.loc[df["max_tariff_savings"].idxmax()]
                max_pc = df_max["continuous_flex_pct"]
                max_uptime = df_max["system_uptime_pct"]

                max_pc = min([max_pc, 99])  # if it is 100% push it to 99% to make it easier to visualize
                max_uptime = min([max_uptime, 99])  # if it is 100% push it to 4% to make it easier to visualize

                max_flex = np.array([max_pc, max_uptime]) / 100

            # overlay points
            overlay_shapes = ["s", "^", "P", "o"]
            for i, point in enumerate(overlay_points):
                # calculate the color based on the index
                color = overlay_colors[i % len(overlay_colors)]
                if i == 0 and idx == 3:
                    # use the max point for the tariff overlay
                    point = max_flex
                # scatter the points
                a.scatter(
                    point[0] * 100, 
                    point[1] * 100, 
                    marker=overlay_shapes[i],
                    edgecolor="black",
                    linewidth=1,
                    color=color, 
                    s=100, 
                    clip_on=True
                )

    # add a pad between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # align the labels
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    fig.align_xlabels(ax[0, :])
    fig.align_xlabels(ax[1, :])

    if save:
        for fig_fmt in ["png", "pdf", "svg"]:
            # save the figure
            plt.savefig(
                os.path.join(figpath, "figures/{}/max_savings_contours/{}_month{}.{}".format(fig_fmt,region, month, fig_fmt)),
                dpi=300,
                bbox_inches="tight",
            )

# define the region and month

ALL_REGIONS = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
# REGIONS/MONTHS let a long sweep be resumed a region or month at a time without editing
# this file. Both default to the full grid; the manuscript figure is CAISO month 7, so
# regenerating just that panel is REGIONS=CAISO MONTHS=7
regions = os.environ.get("REGIONS", ",".join(ALL_REGIONS)).split(",")
months = [int(m) for m in os.environ.get("MONTHS", ",".join(str(m) for m in range(1, 13))).split(",")]

# power capacity, uptime
overlay = True
overlay_points = np.array([
    [0.99, 0.04],
    [0.01, 0.25],
    [0.5, 0.5],
    [0.25, 0.99]
])

figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# to run the analysis
for region in regions:
    if GENERATE_DATA:
        # solve every month of this region in one fan-out
        generate_data(region, list(months))
        print(f"{region}: data written for months {list(months)}", flush=True)

    for month in months:
        # read the data from the csv file
        datafile = os.path.join( figpath, "processed_data", "max_savings_contours", f"{region}_month{month}.csv")
        df = pd.read_csv(datafile)

        # generate the figure
        generate_figure(
            df, 
            region, 
            month, 
            overlay_points=overlay_points, 
            overlay=overlay,
            save=True
        )

        # close the figure
        plt.close()