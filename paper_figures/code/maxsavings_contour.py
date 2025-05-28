# imports
from analysis.pricesignal import getmef, getlmp
from analysis.maxsavings import max_mef_savings, max_aef_savings, max_lmp_savings
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# define plotting defaults
plt.rcParams.update({'font.size': 24,
                        'axes.linewidth': 1.5,
                        'lines.linewidth': 2,
                        'lines.markersize': 6,
                        'xtick.major.size': 3,
                        'xtick.major.width': 1,
                        'ytick.major.size': 3,
                        'ytick.major.width': 1,
                        'xtick.direction': 'out',
                        'ytick.direction': 'out',})

# define the region and month
region = "ERCOt"
month = 5

# get the data
mef_data = getmef(region, month)
lmp_data = getlmp(region, month)

# solve max mef savings in parallel for a range of uptime and continuous flex
uptimes = np.arange(0,25,2)/24  # 1 to 24 hours to percent in intervals
continuous_flex = np.arange(0, 1.01, 0.1)  # 0 to 100%

baseload = np.ones_like(mef_data)  # 1MW flat baseline load

# create a container for results
max_mef_savings_results = np.zeros((len(uptimes), len(continuous_flex)))
max_aef_savings_results = np.zeros((len(uptimes), len(continuous_flex)))
max_lmp_savings_results = np.zeros((len(uptimes), len(continuous_flex)))

# calculate max mef savings
for i, uptime in enumerate(uptimes):
    for j, flex in enumerate(continuous_flex):
        max_mef_savings_results[i, j] = max_mef_savings(mef_data, uptime, flex, baseload)
        # max_aef_savings_results[i, j] = max_aef_savings(mef_data, uptime, flex, baseload) # TODO- add AEF data
        max_lmp_savings_results[i, j] = max_lmp_savings(lmp_data, uptime, flex, baseload)
        # max_tariff_savings_results[i, j] = max_tariff_savings(lmp_data, uptime, flex, baseload) # TODO- add tariff data

# create figure - 2x2 grid for MEF[0,0], AEF[1,0], LMP[0,1], Tariffs[1,1] savings 
fig, ax = plt.subplots(2,2,figsize=(18, 14))

clevels = np.arange(0, 40.1, 1)  # levels for contour plots
cmap = "BuGn"
# plot max MEF savings
contour = ax[0,0].contourf(continuous_flex * 100, uptimes * 100, max_mef_savings_results, levels=clevels, cmap=cmap)
cbar = fig.colorbar(contour, ax=ax[0,0])
ax[0,0].set_title('Marginal Emissions')
cbar.set_label('Savings (%)')

# plot max AEF savings
# contour = ax[1,0].contourf(continuous_flex * 100, uptimes * 100, max_aef_savings_results, levels=clevels, cmap=cmap)
# cbar = fig.colorbar(contour, ax=ax[1,0])
ax[1,0].set_title('Average Emissions')
cbar.set_label('Savings (%)')

# plot max LMP savings
contour = ax[0,1].contourf(continuous_flex * 100, uptimes * 100, max_lmp_savings_results, levels=clevels, cmap=cmap)
cbar = fig.colorbar(contour, ax=ax[0,1])
ax[0,1].set_title('Day-ahead Market')
cbar.set_label('Savings (%)')

# plot max Tariff savings
# contour = ax[1,1].contourf(continuous_flex * 100, uptimes * 100, max_tariff_savings_results, levels=clevels, cmap=cmap)
# cbar = fig.colorbar(contour, ax=ax[1,1])
ax[1,1].set_title('Industrial Tariff IDXXXXXX')
cbar.set_label('Savings (%)')

for a in ax.flatten():
    a.set_aspect('equal', adjustable='box')
    a.set(
        xlim=(0, 100),
        xticks=np.arange(10, 101, 20),
        yticks=np.arange(0, 101, 20),
        ylim=(5, 100),
        xlabel='Continuous Flexibility (%)',
        ylabel='System Uptime (%)',
    )

# add a pad between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# align the labels
fig.align_ylabels(ax[:,0])
fig.align_ylabels(ax[:,1])
fig.align_xlabels(ax[0,:])
fig.align_xlabels(ax[1,:])

# add a title
fig.suptitle(f'Max Savings for {region} in Month {month}', fontsize=24, fontweight='bold')

# save figure
figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig.savefig(os.path.join(figpath, "figures/png", f"max_savings_{region}_month{month}.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(figpath, "figures/pdf", f"max_savings_{region}_month{month}.pdf"), dpi=300, bbox_inches='tight')

# flatten into a dataframe
mef = max_mef_savings_results.flatten()
aef =np.zeros_like(mef) # TODO - placeholder for AEF savings
lmp = max_lmp_savings_results.flatten()
tariff = np.zeros_like(mef)  # TODO - placeholder for tariff savings
continuous_flex_flat = np.tile(continuous_flex, len(uptimes))
uptimes_flat = np.repeat(uptimes, len(continuous_flex))

# create a dataframe
df = pd.DataFrame({
    'continuous_flex_pct': continuous_flex_flat*100,
    'system_uptime_pct': uptimes_flat*100,
    'max_mef_savings': mef,
    'max_aef_savings': aef,
    'max_lmp_savings': lmp,
    'max_tariff_savings': tariff
})
# save the dataframe to a csv file
df.to_csv(os.path.join(figpath, "processed_data", f"max_savings_{region}_month{month}.csv"), index=False)