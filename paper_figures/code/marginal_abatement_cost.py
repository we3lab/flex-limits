import pandas as pd
import numpy as np
import calendar, os, json
import matplotlib.pyplot as plt
import calendar
from analysis.pricesignal import getmef, getaef, getdam, gettariff
import analysis.maxsavings as ms 
from analysis.acc_curve import acc_curve
from analysis.overlay_costs import _add_scc_and_rec
from electric_emission_cost import costs
from electric_emission_cost.units import u
from electric_emission_cost import utils


with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

def get_mac(df):
    # get cost optimal point
    cost_optimal = df.loc[df['alignment_fraction'] == 0]
    df = df[df['alignment_fraction'] >=0]

    # for all other points, calc the shadow cost relative to the cost optimal point
    shadow_cost = []
    emissions = []
    for index, row in df.iterrows():
        delta_cost = row['electricity_cost'] - cost_optimal['electricity_cost'].values[0]
        delta_emissions = cost_optimal['emissions'].values[0] - row['emissions']
        sc = max([delta_cost / (delta_emissions + 1e-12), 1e-2])
        shadow_cost.append(sc)
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

region="CAISO"
month=7
year = 2023

# read in data 
basepath =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dfs_wholesale = []
dfs_tariff = []

# calculate baselines 
_, num_days = calendar.monthrange(year, month = month) 
startdate_dt, enddate_dt = ms.get_start_end(month)

baseline_dam = getdam(region, month, basepath).sum()
baseline_mef = getmef(region, month, basepath).sum() / 1000  # convert kg/MWh to mton/kWh
baseline_aef = getaef(region, month, basepath).sum() * num_days / 1000  # convert kg/MWh to mton/kWh

# calculate tariff baseline
tariff = gettariff(region, basepath=basepath, full_list=False) 
baseload = np.ones(num_days*24)
charge_dict = costs.get_charge_dict(startdate_dt, enddate_dt, tariff, resolution="1h")
baseline_tariff, _= costs.calculate_cost(
    charge_dict,
    {"electric":baseload,
    "gas": np.zeros_like(baseload)},
    resolution="1h",
    consumption_estimate=0,
    model=None,
    electric_consumption_units=u.MW
)

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
    emissions_pct = 100*(baseline_mef - emissions) / baseline_mef
    ax[0].plot(emissions_pct, 
                shadow_cost, 
                label=system_name, 
                color=color_map[system_name]
                )
    ax[0].scatter(emissions_pct,
                shadow_cost, 
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=15,
                zorder=3
                ) 

    pareto_tariff = pd.read_csv(basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name))
    shadow_cost, emissions = get_mac(pareto_tariff)
    emissions_pct = 100*(baseline_aef - emissions) / baseline_aef
    ax[1].plot(emissions_pct, shadow_cost, label=system_name, color=color_map[system_name])
    ax[1].scatter(emissions_pct,
                shadow_cost, 
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=15,
                zorder=3
                ) 

# overlay scc line
_add_scc_and_rec(ax[0], regions=['CAISO'], width=0.15, scc=True, rec=False, plot_scc_by="mean", emission_basis='mef')
# ax[0].set_xlim(0.005, 0.01)

# _add_scc_and_rec(ax[1], regions=['CAISO'], width=0.15, scc=True, rec=False, plot_scc_by="mean", emission_basis='aef')
# ax[1].set_xlim(0.14, 0.20)

for a in ax.flatten(): 
    a.set_yscale("log")
    a.set_xlabel("Scope 2 Emissions Savings (%)")
    a.set_ylabel("Abatement Cost ($/ton)")
    a.set_ylim(0.01, 1e5)
ax[0].set_xlim(0., 30)
ax[1].set_xlim(-5, 20)