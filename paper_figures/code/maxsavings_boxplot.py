import os, json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import pricesignal as ps
from analysis import maxsavings as ms

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

# import colors
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
incentive_colors = colors["incentive_colors"]

month_arr = np.arange(1, 13)
tariff_month_arr = [1,7]
regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
generate_results= False

overlay_star_region = "CAISO"
overlay_star_month = 7

figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if generate_results:
    mef_savings_sweep = np.zeros((len(regions), len(month_arr)))
    aef_savings_sweep = np.zeros((len(regions), len(month_arr)))
    dam_savings_sweep = np.zeros((len(regions), len(month_arr)))
    # cannot be np.array because each region has different number of tariffs
    tariff_savings_sweep = []
    
    for i, reg in enumerate(regions):
        region_list = []
        for j, month in enumerate(month_arr):
            print(f"Processing region: {reg}, month: {month}")

            # MEF
            mef = ps.getmef(region=reg, month=month)
            # minimum uptime - 0% forces to 1 timestep
            mef_savings_sweep[i, j] = ms.max_mef_savings(
                data=mef,
                system_uptime=0.0,
                continuous_flex=0.0,
                baseload=np.ones_like(mef) * 1000,
            )

            # AEF
            aef = ps.getaef(region=reg, month=month)
            aef_savings_sweep[i, j] = ms.max_aef_savings(
                data=aef,
                system_uptime=0.0,
                continuous_flex=0.0,
                baseload=np.ones_like(aef) * 1000,
            )

            # DAM
            dam = ps.getdam(region=reg, month=month)
            dam_savings_sweep[i, j] = ms.max_dam_savings(
                data=dam,
                system_uptime=0.0,
                continuous_flex=0.0,
                baseload=np.ones_like(dam) * 1000,
            )

            # Tariff
            tariffs = ps.gettariff(region=reg, full_list=True)
            for tariff in tariffs:
                startdate_dt, enddate_dt = ms.get_start_end(month)
                month_length = int((enddate_dt - startdate_dt) / np.timedelta64(1, "h"))
                try:
                    tariff_savings = ms.max_tariff_savings(
                        data=tariff,
                        system_uptime=0.0,
                        continuous_flex=1.0,
                        baseload=np.ones(month_length) * 1000,
                        startdate_dt=startdate_dt,
                        enddate_dt=enddate_dt,
                        uptime_equality=False
                    )
                    region_list.append(tariff_savings)
                
                except ZeroDivisionError:
                    print(f"ZeroDivisionError in tariff {tariff['label'].values[0]}")

        tariff_savings_sweep.append(region_list)

    # sort the regions by the max savings in either DAM or MEF
    max_savings = np.maximum(mef_savings_sweep.max(axis=1), dam_savings_sweep.max(axis=1))
    sorted_indices = np.argsort(max_savings)[::-1]

    # reorder the savings arrays based on sorted indices
    mef_savings_sweep = mef_savings_sweep[sorted_indices, :]
    aef_savings_sweep = aef_savings_sweep[sorted_indices, :]
    dam_savings_sweep = dam_savings_sweep[sorted_indices, :]
    sorted_tariff_savings_sweep = []
    for i in range(len(tariff_savings_sweep)):
        sorted_tariff_savings_sweep.append(tariff_savings_sweep[sorted_indices[i]])

    # reorder the regions based on sorted indices
    regions = [regions[i] for i in sorted_indices]

    # save data in a pandas DataFrame
    regions_expanded = np.repeat(regions, len(month_arr))
    months_expanded = np.tile(month_arr, len(regions))
    # when we flatten tariff savings it is of different length,
    # so cannot be included in the DataFrame with MEF, AEF, and DAM
    flat_tariff_savings = [
        tariff 
        for region_list in sorted_tariff_savings_sweep 
        for tariffs in region_list
    ]
    tariff_regions = []
    for i, region_list in enumerate(sorted_tariff_savings_sweep):
        tariff_regions += [regions[i]] * len(region_list)

    tariff_savings_data = {
        "region": tariff_regions.flatten(),
        "tariff_savings": flat_tariff_savings
    }
    savings_data = {
        "region": regions_expanded,
        "month": months_expanded,
        "mef_savings": mef_savings_sweep.flatten(),
        "aef_savings": aef_savings_sweep.flatten(),
        "dam_savings": dam_savings_sweep.flatten(),
    }

    pd.DataFrame(savings_data).to_csv(figpath + "/processed_data/savings_limit_boxplot.csv", index=False)
    pd.DataFrame(savings_data).to_csv(figpath + "/processed_data/tariff_savings_limit_boxplot.csv", index=False)

else:
    # read data from file
    savings_data = pd.read_csv(
        os.path.join(figpath, "processed_data", "savings_limit_boxplot.csv")
    )
    tariff_savings_data = pd.read_csv(
        os.path.join(figpath, "processed_data", "tariff_maxsavings_all.csv")
    )

    # group savings_data by region
    regions = savings_data["region"].unique()

    mef_savings_sweep = np.array([savings_data["mef_savings"][savings_data["region"] == r].values for r in regions])
    aef_savings_sweep = np.array([savings_data["aef_savings"][savings_data["region"] == r].values for r in regions])
    dam_savings_sweep = np.array([savings_data["dam_savings"][savings_data["region"] == r].values for r in regions])
    sorted_tariff_savings_sweep = []
    for r in regions:
        tariff_savings = tariff_savings_data[r].values
        sorted_tariff_savings_sweep.append(tariff_savings)


# plot box and whisker
# create a plot of the emissions savings
fig, ax = plt.subplots(figsize=(90/25.4, 70/25.4))
aef_plot = ax.boxplot(
    aef_savings_sweep.T,
    positions=np.arange(len(regions)) - 0.3,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 0.5, "facecolor": incentive_colors["aef"]},
    whiskerprops={"linewidth": 0.5},
    capprops={"linewidth": 0.5},
)                    
                
mef_plot = ax.boxplot(
    mef_savings_sweep.T,
    positions=np.arange(len(regions)) - 0.1,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 0.5, "facecolor": incentive_colors["mef"]},
    whiskerprops={"linewidth": 0.5},
    capprops={"linewidth": 0.5},
)

dam_plot = ax.boxplot(
    dam_savings_sweep.T,
    positions=np.arange(len(regions)) + 0.1,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 0.5, "facecolor": incentive_colors["dam"]},
    whiskerprops={"linewidth": 0.5},
    capprops={"linewidth": 0.5},
)

tariff_plot = ax.boxplot(
    sorted_tariff_savings_sweep,
    positions=np.arange(len(regions)) + 0.3,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 0.5, "facecolor": incentive_colors["tariff"]},
    whiskerprops={"linewidth": 0.5},
    capprops={"linewidth": 0.5},
)

ax.set(
    ylabel="Savings [%]",
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    yticks=np.arange(0, 161, 20),
    ylim=(0, 160),
)

ax.legend(
    [
        aef_plot["boxes"][0],
        mef_plot["boxes"][0], 
        dam_plot["boxes"][0],
        tariff_plot["boxes"][0],
    ],
    ['AEF', 'MEF', 'DAM', 'Tariff'], 
    loc='best',
    frameon=False, 
    fontsize=7, 
    handlelength=0.75, 
    handleheight=0.75,
    handletextpad=0.4,
)

# overlay star
mef_overlay = savings_data["mef_savings"][savings_data["region"] == overlay_star_region][savings_data["month"] == overlay_star_month].values[0]
aef_overlay = savings_data["aef_savings"][savings_data["region"] == overlay_star_region][savings_data["month"] == overlay_star_month].values[0]
dam_overlay = savings_data["dam_savings"][savings_data["region"] == overlay_star_region][savings_data["month"] == overlay_star_month].values[0]
tariff_max_overlay = pd.read_csv(os.path.join(figpath, "processed_data", "max_savings_contours", f"{overlay_star_region}_month{overlay_star_month}.csv"))
tariff_overlay = max(tariff_max_overlay["max_tariff_savings"].values)

region_index = list(regions).index(overlay_star_region)
ax.scatter(region_index - 0.3, aef_overlay, marker="D", color="k", s=6, zorder=5)
ax.scatter(region_index - 0.1, mef_overlay, marker="D", color="k", s=6, zorder=5)
ax.scatter(region_index + 0.1, dam_overlay, marker="D", color="k", s=6, zorder=5)
ax.scatter(region_index + 0.3, tariff_overlay, marker="D", color="k",s=6, zorder=5)

# save figure
figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        os.path.join(figpath, "figures", ext, f"savings_limit_boxplot.{ext}"),
        dpi=300,
        bbox_inches="tight",
    )