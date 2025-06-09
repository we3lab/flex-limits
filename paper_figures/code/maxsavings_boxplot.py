import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import pricesignal as ps
from analysis import maxsavings as ms

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

month_arr = np.arange(1, 13)
regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
mef_savings_sweep = np.zeros((len(regions), len(month_arr)))
aef_savings_sweep = np.zeros((len(regions), len(month_arr)))
dam_savings_sweep = np.zeros((len(regions), len(month_arr)))
tariff_savings_sweep = np.zeros((len(regions), len(month_arr)))

for i, reg in enumerate(regions):
    for j, month in enumerate(month_arr):

        print(f"Processing region: {reg}, month: {month}")

        # MEF
        mef = ps.getmef(region=reg, month=month)
        # minimum uptime - 0% forces to 1 timestep
        mef_savings_sweep[i, j] = ms.max_mef_savings(
            data=mef,
            system_uptime=0.0,
            continuous_flex=0.0,
            baseload=np.ones_like(mef),
        )

        # AEF
        aef = ps.getaef(region=reg, month=month)
        aef_savings_sweep[i, j] = ms.max_aef_savings(
            data=aef,
            system_uptime=0.0,
            continuous_flex=0.0,
            baseload=np.ones_like(aef),
        )

        # DAM
        dam = ps.getdam(region=reg, month=month)
        dam_savings_sweep[i, j] = ms.max_dam_savings(
            data=dam,
            system_uptime=0.0,
            continuous_flex=0.0,
            baseload=np.ones_like(dam),
        )

        # Tariff
        # TODO: remove this if/else once bug is fixed
        if reg == "PJM":
            tariff_savings_sweep[i, j] = 0
        else:
            tariff = ps.gettariff(region=reg)
            startdate_dt, enddate_dt = ms.get_start_end(month)
            month_length = int((enddate_dt - startdate_dt) / np.timedelta64(1, "h"))
            tariff_savings_sweep[i, j] = ms.max_tariff_savings(
                data=tariff,
                system_uptime=0.0,
                continuous_flex=0.0,
                baseload=np.ones(month_length),
                startdate_dt=startdate_dt,
                enddate_dt=enddate_dt,
                uptime_equality=False
            )

# sort the regions by the max savings in either DAM or MEF
max_savings = np.maximum(mef_savings_sweep.max(axis=1), dam_savings_sweep.max(axis=1))
sorted_indices = np.argsort(max_savings)[::-1]

# reorder the savings arrays based on sorted indices
mef_savings_sweep = mef_savings_sweep[sorted_indices, :]
aef_savings_sweep = aef_savings_sweep[sorted_indices, :]
dam_savings_sweep = dam_savings_sweep[sorted_indices, :]
tariff_savings_sweep = tariff_savings_sweep[sorted_indices, :]

# reorder the regions based on sorted indices
regions = [regions[i] for i in sorted_indices]
# aef_savings_sweep and tariff_savings_sweep are not used in this plot, but can be calculated similarly if needed

# plot box and whisker
# create a plot of the emissions savings
fig, ax = plt.subplots(figsize=(10, 6))
aef_plot = ax.boxplot(
    aef_savings_sweep.T,
    positions=np.arange(len(regions)) - 0.3,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 1.5, "facecolor": "royalblue"},
    whiskerprops={"linewidth": 1.5},
    capprops={"linewidth": 1.5},
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
    boxprops={"linewidth": 1.5, "facecolor": "darkseagreen"},
    whiskerprops={"linewidth": 1.5},
    capprops={"linewidth": 1.5},
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
    boxprops={"linewidth": 1.5, "facecolor": "palegoldenrod"},
    whiskerprops={"linewidth": 1.5},
    capprops={"linewidth": 1.5},
)

tariff_plot = ax.boxplot(
    tariff_savings_sweep.T,
    positions=np.arange(len(regions)) + 0.3,
    widths=0.15,
    tick_labels=regions,
    patch_artist=True,
    showfliers=False,
    whis=(0, 100),
    medianprops={"linewidth": 0},
    boxprops={"linewidth": 1.5, "facecolor": "violet"},
    whiskerprops={"linewidth": 1.5},
    capprops={"linewidth": 1.5},
)

ax.set(
    ylabel="Savings [%]",
    title="Maximum savings from flexibility",
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    yticks=np.arange(0, 161, 20),
    ylim=(0, 160),
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="center")

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
    fontsize=20, 
    handlelength=1, 
    handleheight=1
)

# save figure
figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig.savefig(
    os.path.join(figpath, "figures/png", f"savings_limit_boxplot.png"),
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(figpath, "figures/pdf", f"savings_limit_boxplot.pdf"),
    dpi=300,
    bbox_inches="tight",
)

# save data in a pandas DataFrame
regions_expanded = np.repeat(regions, len(month_arr))
months_expanded = np.tile(month_arr, len(regions))
savings_data = {
    "region": regions_expanded,
    "month": months_expanded,
    "mef_savings": mef_savings_sweep.flatten(),
    "aef_savings": aef_savings_sweep.flatten(),
    "dam_savings": dam_savings_sweep.flatten(),
    "tariff_savings": tariff_savings_sweep.flatten()}

pd.DataFrame(savings_data).to_csv(figpath + "/processed_data/savings_limit_boxplot.csv", index=False)