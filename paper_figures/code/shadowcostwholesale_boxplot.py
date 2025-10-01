import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from analysis.overlay_costs import _add_scc_and_rec
from analysis import pricesignal as ps
from analysis import emissionscost as ec
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
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]


generate_data = False


# The order of regions has been updated to reflect the desired processing sequence.
regions = ["SPP", "CAISO", "ERCOT",  "PJM", "MISO", "NYISO", "ISONE"]

systems = {
    "maxflex" : {
        "system_uptime": 1/24,  # minimum uptime
        "continuous_flexibility": 1.0 # full flexibility
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0  
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5 
    },
    "100uptime_25flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.25 
    },
}

emissions_type="mef"

paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if generate_data == True:
    for region in regions:
        for system_name, params in systems.items():
            results_dicts = []
            results_dicts_50 = []
            for month in range(1,13):
                tmp = ec.shadowcost_wholesale(region=region,
                                                    month=month,
                                                    system_uptime=params['system_uptime'],
                                                    continuous_flexibility=params['continuous_flexibility'],
                                                    emissions_type=emissions_type)

                results_dicts.append(tmp)

                tmp50 = ec.shadowcost_wholesale(region=region,
                                                    month=month,
                                                    system_uptime=params['system_uptime'],
                                                    continuous_flexibility=params['continuous_flexibility'],
                                                    emissions_type=emissions_type,
                                                    abatement_fraction=0.50)
                results_dicts_50.append(tmp50)
                                                    
            
            # collapse the list of dicts into a dict of lists 
            results_df = pd.DataFrame(results_dicts)
            results_df_50 = pd.DataFrame(results_dicts_50)

            # add a months columns and rearrange the columns to have it first
            results_df["month"] = range(1,13) # add a col for month
            cols = ["month"] + [col for col in results_df.columns if col != "month"]  
            results_df = results_df[cols]  

            results_df_50["month"] = range(1,13) # add a col for month
            results_df_50 = results_df_50[cols]

            # put the df in the params dict
            params["results"] = results_df

            # save df in results folder as a csv file
            results_df.to_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_wholesale_{emissions_type}", f"{region}_{system_name}.csv"), index=False)
            results_df_50.to_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_wholesale_{emissions_type}", f"{region}_{system_name}_50pct_abatement.csv"), index=False)
else:
    ##
    pass

# Plotting the results
fig, ax = plt.subplots(figsize=(180 / 25.4, 45 / 25.4))

width = 0.15

offset = [-width * 1.6, -width * 0.55, width * 0.55, width * 1.6]  # Offset for each system's shadow price for better visibility
num_systems = len(systems)  # Number of systems to plot
system_names = list(systems.keys())
color_map = {k: sys_colors[k] for k in system_names if k in sys_colors}


for region_idx, region in enumerate(regions):
    # Iterate over each system and plot the results
    idx = 0 
    for sys in system_names:
        # Load the results for the current region and system 
        results_df = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_wholesale_{emissions_type}", f"{region}_{sys}.csv"))
        results_df_50 = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_wholesale_{emissions_type}", f"{region}_{sys}_50pct_abatement.csv"))
        
        # Plot the shadow price as a bar plot
        bottom = results_df["shadow_price_usd_ton"].min()
        height = results_df["shadow_price_usd_ton"].max() - bottom

        ax.bar(region_idx + offset[idx], 
                    height = height, 
                    bottom=bottom, 
                    width=width, 
                    label=sys, 
                    facecolor=color_map[sys],
                    edgecolor='k',
                    linewidth=1,
                    alpha=1, 
                    zorder=2)
        
        # Scatter plot for the shadow price for each month
        ax.scatter(np.ones(len(results_df)) * region_idx + offset[idx], results_df["shadow_price_usd_ton"].values, s=0.1, alpha=0.5, color='k',zorder=4)


        # Add a rectangle for the 50% abatement cost
        bottom_50 = results_df_50["shadow_price_usd_ton"].min()
        height_50 = results_df_50["shadow_price_usd_ton"].max() - bottom_50

        ax.bar(region_idx + offset[idx],
                    height = height_50, 
                    bottom=bottom_50, 
                    width=width, 
                    label=f"{sys} 50% abatement", 
                    facecolor=color_map[sys],
                    edgecolor='k',
                    linewidth=1,
                    alpha=0.5,
                    zorder=1)  # Behind the main bars
        ax.scatter(np.ones(len(results_df_50)) * region_idx + offset[idx], results_df_50["shadow_price_usd_ton"].values, s=0.1, alpha=0.5, color='k',zorder=4)
        idx += 1

ax.set(
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    xlim=(-0.5, len(regions) - 0.5),
    ylabel="Cost of Abatement (USD/ton CO$_2$)",
    ylim=(1e-2, 1e5),
    yticks=np.logspace(-2, 5, num=8),  # Logarithmic scale for y-axis
    yticklabels=[f"{int(10**i):,}" for i in range(-2, 6)],
    yscale="log"
)
ax.set_ylabel("Cost of Abatement (USD/ton CO$_2$)",labelpad=-1)

ax.text(-0.1, 1.06, 'a.', transform=ax.transAxes,
        fontsize=7, fontweight='bold', va='top', ha='left')


# Comment this line out out to disable SCC and REC overlays
_add_scc_and_rec(
    ax, regions=regions, width=width, scc=True, rec=True, plot_scc_by="mean", emission_basis=emissions_type
)

fig.savefig(os.path.join(paperfigs_basepath, "figures/png", f"shadowcost_wholesale_boxplot_{emissions_type}.png"),
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(paperfigs_basepath, "figures/pdf", f"shadowcost_wholesale_boxplot_{emissions_type}.pdf"),
    dpi=300,
    bbox_inches="tight",
)
