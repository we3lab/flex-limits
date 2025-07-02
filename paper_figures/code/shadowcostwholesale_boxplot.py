import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from analysis import pricesignal as ps
from analysis import emissionscost as ec
from analysis import maxsavings as ms

# define plotting defaults
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.linewidth": 2,
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

generate_data = False


regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]

systems = {
    "maxflex" : {
        "system_uptime": 1/24,  # minimum uptime
        "continuous_flexibility": 0.0
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0  
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5 
    },
    "100uptime_75flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.75 
    },
}

paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if generate_data == True:
    for region in regions:
        for system_name, params in systems.items():
            results_dicts = []
            for month in range(1,13):
                tmp = ec.shadowcost_wholesale(region=region,
                                                    month=month,
                                                    system_uptime=params['system_uptime'],
                                                    continuous_flexibility=params['continuous_flexibility'])

                results_dicts.append(tmp)
            
            # collapse the list of dicts into a dict of lists 
            results_df = pd.DataFrame(results_dicts)

            # add a months columns and rearrange the columns to have it first
            results_df["month"] = range(1,13) # add a col for month
            cols = ["month"] + [col for col in results_df.columns if col != "month"]  
            results_df = results_df[cols]  

            # put the df in the params dict
            params["results"] = results_df

            # save df in results folder as a csv file
            results_df.to_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_wholesale", f"{region}_{system_name}.csv"), index=False)
else:
    pass

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

width = 0.15

offset = [-width * 1.6, -width * 0.55, width * 0.55, width * 1.6]  # Offset for each system's shadow price for better visibility
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Colors for each system


ax.set_xlabel("Region")
ax.set_ylabel("Wholesale Shadow Price (USD/metric ton CO2)")

num_systems = len(systems)  # Number of systems to plot

system_names = list(systems.keys())

for region_idx, region in enumerate(regions):
    # Iterate over each system and plot the results
    idx = 0 
    for sys in system_names:
        # Load the results for the current region and system 
        results_df = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_wholesale", f"{region}_{sys}.csv"))
        
        # Plot the shadow price as a bar plot
        bottom = results_df["shadow_price_usd_ton"].min()
        height = results_df["shadow_price_usd_ton"].max() - bottom

        ax.bar(region_idx + offset[idx], 
                    height = height, 
                    bottom=bottom, 
                    width=width, 
                    label=sys, 
                    facecolor=colors[idx],
                    edgecolor='k',
                    linewidth=1.5,
                    alpha=0.5)
        
        # Scatter plot for the shadow price for each month
        ax.scatter(np.ones(len(results_df)) * region_idx + offset[idx], results_df["shadow_price_usd_ton"].values, s=15, color='k')
        idx += 1

ax.set(
    xlabel="Region",
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    ylabel="Cost of Abatement (USD/ton CO2)",
    ylim=(1e-2, 1e5),
    yscale="log"
)

def _add_scc_and_rec(ax, regions, ps, width, scc=True, rec=True):
    values_df = pd.read_csv("data/rec/values.csv") # TODO: update with more regional data
    rec_price_usd_mw = values_df[values_df['type'] == 'rec']['typical'].values[0]  # $/MW
    scc_price_usd_ton = values_df[values_df['type'] == 'scc']['typical'].values[0]  # $/ton
    
    legend_info = {'handles': [], 'labels': []}
    
    if scc:  # Add SCC line
        scc_line = ax.axhline(
            y=scc_price_usd_ton, color='black', linestyle='-', alpha=0.7, linewidth=2,
            label=f'Social Cost of Carbon (${scc_price_usd_ton}/ton)', zorder=0
        )
        legend_info['handles'].append(scc_line)
        legend_info['labels'].append(f'Social Cost of Carbon (${scc_price_usd_ton}/ton)')

    if rec:  # Add REC boxes
        rec_prices_by_region = {}
        for region_idx, region in enumerate(regions):
            # Gather MEF values (kg/MWh) for this ISO across months and hours
            all_mef_kg_per_mwh = []
            for month in range(1, 13):
                mef_data = ps.getmef(region, month) # for 24 hr
                all_mef_kg_per_mwh.extend(mef_data)
            all_mef_ton_per_mwh = np.array(all_mef_kg_per_mwh) / 1000  # kg to ton
            
            # If a wholesaler has a fixed REC price, then
            # the relative value in $/kg varies each hour
            # during the year based on MEF
            # REC price in $/MWh = $/MW * 1 hr
            # REV value in $/kg = ($/MWh) / (kg/MWh) = $/kg
            rec_prices_region = rec_price_usd_mw / all_mef_ton_per_mwh  # $/ton
            rec_prices_region = rec_prices_region[~np.isnan(rec_prices_region)]  # remove nans
            rec_prices_by_region[region] = rec_prices_region
            min_rec = np.min(rec_prices_region)
            max_rec = np.max(rec_prices_region)
            
            # Rectangle spanning bars for ISO
            rec_rect = Rectangle(
                (region_idx - width*2.5, min_rec),  # x, y (left edge, bottom)
                width*5,  # width to cover all bars
                max_rec - min_rec,  # height
                facecolor='lightgray',
                edgecolor='grey',
                alpha=0.5,
                zorder=0,  # behind the bars
                label=f'REC Price $({rec_price_usd_mw}/MW)' if region_idx == 0 else None
            )
            ax.add_patch(rec_rect)
        rec_patch = Patch(
            facecolor='lightgray', alpha=0.5, edgecolor='grey', 
            label=f'REC Price $({rec_price_usd_mw}/MW)'
            )
        legend_info['handles'].append(rec_patch)
        legend_info['labels'].append(f'REC Price $({rec_price_usd_mw}/MW)')
    ax.legend(legend_info['handles'], legend_info['labels'], loc='upper right', frameon=False, fontsize=14)

# Comment this line out out to disable SCC and REC overlays
_add_scc_and_rec(
    ax, regions=regions, ps=ps, width=width, scc=True, rec=True
)

fig.savefig(os.path.join(paperfigs_basepath, "figures/png", "shadowcost_wholesale_boxplot.png"),
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(paperfigs_basepath, "figures/pdf", "shadowcost_wholesale_boxplot.pdf"),
    dpi=300,
    bbox_inches="tight",
)
