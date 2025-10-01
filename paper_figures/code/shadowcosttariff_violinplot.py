import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
import seaborn as sns
import time
from analysis import pricesignal as ps
from analysis import emissionscost as ec
from joblib import Parallel, delayed

from paper_figures.code.shadowcostwholesale_boxplot import _add_scc_and_rec

# VB parameter settings 
systems = {
    "maxflex" : {
        "system_uptime": 0.,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "uptime_equality":False
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.,
        "uptime_equality":True  
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "uptime_equality":True 

    },
    "100uptime_25flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.25, 
        "uptime_equality": True
    },
}

with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

# grid parameter settings 
months = [1,7]
regions = ["SPP", "CAISO", "ERCOT",  "PJM", "MISO", "NYISO", "ISONE"]
emissions_type = "mef"

# data/figure gen settings 
generate_data = True
threads = 20 

paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_data_single(system_name, params, region, month):
    # create a new dataframe for all tariffs in the specified region 
    region_tariffs, region_tariff_ids = ps.gettariff(region=region, full_list=True, return_ids=True)

    results_dicts = []
    for n, tariff in enumerate(region_tariffs):
        try: 
            # calculate shadow cost for the given case (system, tariff, month)
            tmp = ec.shadowcost_tariff(region=region,
                                                tariff_data=tariff, 
                                                month=month,
                                                system_uptime=params["system_uptime"],
                                                continuous_flexibility=params["continuous_flexibility"], 
                                                uptime_equality=params["uptime_equality"], 
                                                threads = 4,
                                                emissions_type=emissions_type,
                                                )

        except ZeroDivisionError: # skipping over tariffs with zero division error in pct_savings calculation - todo: debug this error 

            tmp = {
                "shadow_price_usd_ton": np.nan,
                "cost_optimal_cost_usd": np.nan,
                "emissions_optimal_cost_usd": np.nan,
                "cost_optimal_emissions_ton": np.nan,
                "emissions_optimal_emissions_ton": np.nan
            }

        tmp["region"] = region
        tmp["system"] = system_name 
        tmp["month"] = month 
        tmp["tariff_id"] = region_tariff_ids[n]
        results_dicts.append(tmp)              
    sys_results = pd.DataFrame(results_dicts)
    # rearrange the columns
    cols_first = ["region","system","month", "tariff_id"]
    cols = cols_first + [col for col in sys_results.columns if col not in cols_first]  
    sys_results = sys_results[cols]  
    return sys_results

if generate_data == True:
    for region in regions:
        tasks = []                
        for system_name, params in systems.items():        
            for month in months:
                # create a task for each system, region, and month using delayed
                tasks.append(delayed(generate_data_single)(system_name, params, region, month))
    
        sys_results_df = pd.concat(Parallel(n_jobs=threads, backend="loky")(tasks))
        sys_results_df.to_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_tariff_{emissions_type}", f"{region}.csv"), index=False)


# read in results
results_list = []
sys_list = []
region_list = []
for region_idx, region in enumerate(regions):
    # Load the results for the current region and system 
    sys_results_df = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", f"shadowcost_tariff_{emissions_type}", f"{region}.csv"))
    results_list.append(sys_results_df)

# create combined data frame                             
results_df = pd.concat(results_list, ignore_index=True)
results_df.dropna(how = "any", inplace=True)
results_df = results_df[results_df.shadow_price_usd_ton > -1e-8]  # filter out negative shadow prices - error in convex approximation
results_df.loc[results_df.shadow_price_usd_ton < 1e-2, "shadow_price_usd_ton"] = 1e-2 #clip very small/negative prices (necessary for log scale)


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

# create color map 
num_systems = len(systems)  # Number of systems to plot
system_names = list(systems.keys())
color_map = {k: sys_colors[k] for k in system_names if k in sys_colors}

# create figure 
fig, ax = plt.subplots(figsize=(180 / 25.4, 45 / 25.4))

_add_scc_and_rec(ax, regions, width=0.15, scc=True, rec=True, plot_scc_by="mean", emission_basis=emissions_type)

# violin plots 
p1 = sns.violinplot(data=results_df, x = "region", y ="shadow_price_usd_ton", hue = "system", 
                    gap = 0.4, inner = "point", density_norm="width", 
                    inner_kws= {"s": 0.1, "alpha":0.5,}, alpha = 1.0, 
                   palette=color_map, log_scale=True, split=True,  ax=ax, cut=0, common_norm=True)

ax.set(
    xlabel="",
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    xlim=(-0.5, len(regions) - 0.5),
    ylim=(1e-2, 1e5),
    yticks=np.logspace(-2, 5, num=8),
    yscale="log"
)
ax.set_ylabel("Cost of Abatement (USD/ton CO$_2$)",labelpad=-1)
# rotate x-axis labels
# plt.setp(ax.get_xticklabels(), rotation=45, ha="center")

ax.text(-0.1, 1.06, 'b.', transform=ax.transAxes,
        fontsize=7, fontweight='bold', va='top', ha='left')

# create legend 
handles, _ = p1.get_legend_handles_labels()
system_titles = ["Maximum Savings", "25% Uptime, 0% Power Capacity", "50% Uptime, 50% Power Capacity", "100% Uptime, 25% Power Capacity"]
# ax.legend(handles, system_titles, ncol = 1, loc = (0.15, -0.48), frameon = False)
ax.get_legend().remove()


# save figure 
for figure_type in ["png","svg", "pdf"]: 
    fig.savefig(os.path.join(paperfigs_basepath, "figures", figure_type, f"shadowcost_tariff_violinplot_{emissions_type}." + figure_type),
        dpi=300,
        bbox_inches="tight",
    )
