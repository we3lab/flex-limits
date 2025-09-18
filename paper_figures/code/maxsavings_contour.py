# imports
import os, json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.pricesignal import getmef, getdam, getaef, gettariff
from analysis.maxsavings import (
    max_mef_savings, 
    max_aef_savings, 
    max_dam_savings, 
    max_tariff_savings, 
    get_start_end
)
GENERATE_DATA = True
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

def generate_data(region, month):
    """
    Generate the maximum savings contours for a given region and month.
    """
    # get the data
    mef_data = getmef(region, month)
    aef_data = getaef(region, month)
    dam_data = getdam(region, month)
    tariff_data = gettariff(region, full_list=False)

    # solve max mef savings in parallel for a range of uptime and continuous flex
    uptimes = np.arange(0, 25, 1) / 24  # 1 to 24 hours to percent in intervals
    continuous_flex = np.arange(0, 1.01, 0.1)  # 0 to 100%

    non_tariff_baseload = np.ones_like(mef_data)  # 1MW flat baseline load

    # tariffs require a full month of data to properly incorporate monthly demand charges
    startdate_dt, enddate_dt = get_start_end(month)
    month_length = int((enddate_dt - startdate_dt) / np.timedelta64(1, "h"))
    tariff_baseload = np.ones(month_length)

    # create a container for results
    max_mef_savings_results = np.zeros((len(uptimes), len(continuous_flex)))
    max_aef_savings_results = np.zeros((len(uptimes), len(continuous_flex)))
    max_dam_savings_results = np.zeros((len(uptimes), len(continuous_flex)))
    max_tariff_savings_results = np.zeros((len(uptimes), len(continuous_flex)))

    # calculate max mef savings
    for i, uptime in enumerate(uptimes):
        for j, flex in enumerate(continuous_flex):
            max_mef_savings_results[i, j] = max_mef_savings(
                mef_data, uptime, flex, non_tariff_baseload
            )
            max_aef_savings_results[i, j] = max_aef_savings(
                aef_data, uptime, flex, non_tariff_baseload
            )
            max_dam_savings_results[i, j] = max_dam_savings(
                dam_data, uptime, flex, non_tariff_baseload
            )
            max_tariff_savings_results[i, j] = max_tariff_savings(
                tariff_data, 
                uptime, 
                flex, 
                tariff_baseload, 
                uptime_equality=True, 
                startdate_dt=startdate_dt, 
                enddate_dt=enddate_dt
            )
    # flatten into a dataframe
    mef = max_mef_savings_results.flatten()
    aef = max_aef_savings_results.flatten()
    dam = max_dam_savings_results.flatten()
    tariff = max_tariff_savings_results.flatten()
    continuous_flex_flat = np.tile(continuous_flex, len(uptimes))
    uptimes_flat = np.repeat(uptimes, len(continuous_flex))

    # create a dataframe
    figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df = pd.DataFrame(
        {
            "continuous_flex_pct": continuous_flex_flat * 100,
            "system_uptime_pct": uptimes_flat * 100,
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
    return df

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

# regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
# months = np.arange(1, 13)

regions = ["CAISO"]
months = [7]

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
    for month in months:
        if GENERATE_DATA:
            # generate the data
            df = generate_data(region, month)
            datafile = os.path.join( figpath, "processed_data", "max_savings_contours", f"{region}_month{month}.csv")
            df.to_csv(datafile)

        else:
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