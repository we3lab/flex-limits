import pandas as pd
import numpy as np
import os, json
import matplotlib.pyplot as plt
from paper_figures.code.shadowcostwholesale_boxplot import _add_scc_and_rec

with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

def get_mac(df):
    # get cost optimal point
    cost_optimal = df.loc[df['alignment_fraction'] == 0]
    df = df[df['alignment_fraction'] > 0]

    # for all other points, calc the shadow cost relative to the cost optimal point
    shadow_cost = []
    emissions = []
    for index, row in df.iterrows():
        delta_cost = row['electricity_cost'] - cost_optimal['electricity_cost'].values[0]
        delta_emissions = cost_optimal['emissions'].values[0] - row['emissions']
        shadow_cost.append(delta_cost / (delta_emissions + 1e-12))
        emissions.append(row['emissions'])

    return shadow_cost, emissions

    
systems = {
    "maxflex" : {
        "system_uptime": 0.0,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "uptime_equality": False,
        "pareto_stepsize": 0.05
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.05
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.1 
    },
    "100uptime_25flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.25, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.1 
    },
}

# read in data 
basepath =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dfs_wholesale = []
dfs_tariff = []

# define plotting defaults
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 1,
        "lines.linewidth": 1.5,
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
fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 80 / 25.4), layout="tight")


for sys in systems.keys(): 
    system_name = sys
    pareto_wholesale = pd.read_csv(basepath + "/paper_figures/processed_data/pareto_wholesale/pareto_front_wholesale_{}.csv".format(system_name))
    shadow_cost, emissions = get_mac(pareto_wholesale)
    ax[0].plot(emissions, 
                shadow_cost, 
                label=system_name, 
                color=color_map[system_name]
                )
    ax[0].scatter(emissions,
                shadow_cost, 
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=15,
                zorder=3
                ) 

    pareto_tariff = pd.read_csv(basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name))
    shadow_cost, emissions = get_mac(pareto_tariff)
    ax[1].plot(emissions, shadow_cost, label=system_name, color=color_map[system_name])
    ax[1].scatter(emissions,
                shadow_cost, 
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=15,
                zorder=3
                ) 

# overlay scc line
_add_scc_and_rec(ax[0], regions=['CAISO'], width=0.15, scc=True, rec=True, plot_scc_by="mean", emission_basis='mef')
ax[0].set_xlim(0.005, 0.01)

_add_scc_and_rec(ax[1], regions=['CAISO'], width=0.15, scc=True, rec=True, plot_scc_by="mean", emission_basis='aef')
ax[1].set_xlim(0.14, 0.20)

for a in ax.flatten(): 
    a.set_yscale("log")
    a.set_xlabel("Emissions (ton)")
    a.set_ylabel("Abatement Cost ($/ton)")
    a.set_ylim(0.01, 1e5)