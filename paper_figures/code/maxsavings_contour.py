# imports
import os
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

# define plotting defaults
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "xtick.major.size": 3,
        "xtick.major.width": 1,
        "ytick.major.size": 3,
        "ytick.major.width": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

# define the region and month

regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
months = np.arange(1, 13)

for region in regions:
    for month in months:
        # get the data
        mef_data = getmef(region, month)
        aef_data = getaef(region, month)
        dam_data = getdam(region, month)
        tariff_data = gettariff(region, full_list=False)

        # solve max mef savings in parallel for a range of uptime and continuous flex
        uptimes = np.arange(0, 25, 2) / 24  # 1 to 24 hours to percent in intervals
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
                    startdate_dt=startdate_dt, 
                    enddate_dt=enddate_dt,
                    uptime_equality=True
                )

        # create figure - 2x2 grid for MEF[0,0], AEF[1,0], DAM[0,1], Tariffs[1,1] savings
        fig, ax = plt.subplots(2, 2, figsize=(18, 14))

        # plot max MEF savings
        contour = ax[0, 0].contourf(
            continuous_flex * 100,
            uptimes * 100,
            max_mef_savings_results,
            levels=np.arange(0,30.1, 2.5),
            extend="max",
            cmap="Greens",
        )
        cbar = fig.colorbar(contour, ax=ax[0, 0])
        ax[0, 0].set_title("Marginal Emissions")
        cbar.set_label("Savings (%)")

        # plot max AEF savings
        contour = ax[1,0].contourf(
            continuous_flex * 100, 
            uptimes * 100, 
            max_aef_savings_results, 
            levels=np.arange(0,30.1, 0.5),
            vmax=30,
            extend="max",
            cmap="Blues",
        )
        cbar = fig.colorbar(contour, ax=ax[1,0])
        ax[1, 0].set_title("Average Emissions")
        cbar.set_label("Savings (%)")

        # plot max DAM savings
        contour = ax[0, 1].contourf(
            continuous_flex * 100,
            uptimes * 100,
            max_dam_savings_results,
            extend="max",
            levels=np.arange(0,100.1, 2.5),
            cmap="YlOrBr",
        )
        cbar = fig.colorbar(contour, ax=ax[0, 1])
        ax[0, 1].set_title("Day-ahead Prices")
        cbar.set_label("Savings (%)")

        # plot max Tariff savings
        contour = ax[1,1].contourf(
            continuous_flex * 100, 
            uptimes * 100, 
            max_tariff_savings_results, 
            extend="max", 
            levels=np.arange(0, 15.1, 1.5),
            cmap="PuRd",
        )
        cbar = fig.colorbar(contour, ax=ax[1,1])
        ax[1, 1].set_title("Sample Tariff: {region}")
        cbar.set_label("Savings (%)")

        for a in ax.flatten():
            a.set_aspect("equal", adjustable="box")
            a.set(
                xlim=(0, 100),
                xticks=np.arange(10, 101, 20),
                yticks=np.arange(0, 101, 20),
                ylim=(5, 100),
                xlabel="Continuous Flexibility (%)",
                ylabel="System Uptime (%)",
            )

        # add a pad between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # align the labels
        fig.align_ylabels(ax[:, 0])
        fig.align_ylabels(ax[:, 1])
        fig.align_xlabels(ax[0, :])
        fig.align_xlabels(ax[1, :])

        # add a title
        fig.suptitle(
            f"Max Savings for {region} in Month {month}", fontsize=24, fontweight="bold"
        )

        # save figure
        figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fig.savefig(
            os.path.join(figpath, "figures/png/max_savings_contours", f"{region}_month{month}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(figpath, "figures/pdf/max_savings_contours", f"{region}_month{month}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

        # flatten into a dataframe
        mef = max_mef_savings_results.flatten()
        aef = max_aef_savings_results.flatten()
        dam = max_dam_savings_results.flatten()
        tariff = max_tariff_savings_results.flatten()
        continuous_flex_flat = np.tile(continuous_flex, len(uptimes))
        uptimes_flat = np.repeat(uptimes, len(continuous_flex))

        # create a dataframe
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
