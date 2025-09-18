import calendar, os, json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import analysis.maxsavings as ms 
from analysis.pricesignal import getmef, getaef, getdam, gettariff
from analysis.acc_curve import acc_curve
from analysis.overlay_costs import _add_scc_and_rec
from electric_emission_cost import costs
from electric_emission_cost.units import u
from electric_emission_cost import utils

# import color maps as json
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

def get_mac(df):
    # get cost optimal point
    cost_optimal = df.loc[df['alignment_fraction'] == 0]
    df = df[df['alignment_fraction'] >0]

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

# VB parameter settings 
systems = {
    "maxflex" : {
        "system_uptime": 0.0,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "uptime_equality": False,
        "pareto_stepsize": 0.05, 
        "marker_shape": "s"
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.05,
        "marker_shape": "^"

    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.1,
        "marker_shape": "P"
    },
    "100uptime_25flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.25, 
        "uptime_equality": True, 
        "pareto_stepsize": 0.1,
        "marker_shape": "o"
    },
}

# grid parameter settings 
region = "CAISO"
month = 7
year = 2023

# data/figure gen settings 
generate_data = False
threads = 20

basepath =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# get date info 
_, num_days = calendar.monthrange(year, month = month) 
startdate_dt, enddate_dt = ms.get_start_end(month)

if generate_data == True: 

    # get wholesale energy market + marginal emissions 
    dam = getdam(region, month, basepath)
    mef = getmef(region, month, basepath)

    # get tariff + average emissions 
    tariff = gettariff(region, basepath=basepath)
    aef = getaef(region, month, basepath)   
    aef = np.tile(aef, num_days) 

    # generate pareto curves for each system 
    pareto_wholesale_list = []
    pareto_tariff_list = []
    for system_name, system_dict in systems.items(): 

        system_uptime = systems[system_name]["system_uptime"]
        continuous_flexibility = systems[system_name]["continuous_flexibility"]
        pareto_stepsize = systems[system_name]["pareto_stepsize"]
        uptime_equality = systems[system_name]["uptime_equality"]

        # dam and mef pareto front 
        baseload = np.ones_like(mef) # 1 MW baseload

        acc = acc_curve(
            baseload=baseload, 
            min_onsteps=max(int(len(mef) * (system_uptime)), 1), 
            flex_capacity = continuous_flexibility, 
            emissions_signal=mef, 
            emissions_type = "mef", 
            cost_signal = dam, 
            costing_type = "dam", 
            startdate_dt= startdate_dt, 
            enddate_dt=enddate_dt,
            uptime_equality = uptime_equality
            )

        em_opt = acc.calc_emissions_optimal(threads = threads)
        cost_opt = acc.calc_cost_optimal(threads = threads)
        savepath = basepath + "/paper_figures/processed_data/pareto_wholesale/pareto_front_wholesale_{}.csv".format(system_name)
        pareto_sys_df = acc.build_pareto_front(stepsize = pareto_stepsize, threads = threads, savepath = savepath)
        pareto_sys_df["system"] = system_name 
        pareto_wholesale_list.append(pareto_sys_df)


        # tariff and aef pareto front 
        baseload = np.ones_like(aef) # 1 MW baseload

        acc = acc_curve(
            baseload=baseload, 
            min_onsteps=max(int(len(aef) * (system_uptime)), 1), 
            flex_capacity = continuous_flexibility, 
            emissions_signal=aef, 
            emissions_type = "aef", 
            cost_signal = tariff, 
            costing_type = "tariff", 
            startdate_dt= startdate_dt, 
            enddate_dt=enddate_dt,
            uptime_equality = uptime_equality
            )
        
        em_opt = acc.calc_emissions_optimal()
        cost_opt = acc.calc_cost_optimal()
        savepath = basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name)
        pareto_sys_df = acc.build_pareto_front(stepsize = pareto_stepsize, savepath = savepath)
        pareto_sys_df["system"] = system_name 
        pareto_tariff_list.append(pareto_sys_df)

# read saved data 
pareto_wholesale_list = []
for system_name, _ in systems.items(): 
    savepath = basepath + "/paper_figures/processed_data/pareto_wholesale/pareto_front_wholesale_{}.csv".format(system_name)
    pareto_sys_df = pd.read_csv(savepath)
    pareto_sys_df["system"] = system_name 
    pareto_wholesale_list.append(pareto_sys_df)

pareto_tariff_list = []
for system_name, _ in systems.items(): 
    savepath = basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name)
    pareto_sys_df = pd.read_csv(savepath)
    pareto_sys_df["system"] = system_name 
    pareto_tariff_list.append(pareto_sys_df)

# combine system dfs
pareto_wholesale_df = pd.concat(pareto_wholesale_list).reset_index(drop = True)
pareto_tariff_df = pd.concat(pareto_tariff_list).reset_index(drop = True)

# calculate wholesale emissions/costs for entire month 
pareto_wholesale_df["emissions"] *= num_days
pareto_wholesale_df["electricity_cost"] *=num_days

# calculate baselines 
baseline_dam = getdam(region, month, basepath).sum() * num_days
baseline_mef = getmef(region, month, basepath).sum() * num_days / 1000   # convert kg/MWh to mton/MWh
baseline_aef = getaef(region, month, basepath).sum() * num_days / 1000   # convert kg/MWh to mton/MWh

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

# convert dollar savings to pct savings based on the baseline
pareto_wholesale_df["electricity_cost_pct"] = 100*(baseline_dam - pareto_wholesale_df["electricity_cost"]) / baseline_dam
pareto_wholesale_df["emissions_pct"] = 100*(baseline_mef - pareto_wholesale_df["emissions"]) / baseline_mef
pareto_tariff_df["electricity_cost_pct"] = 100*(baseline_tariff - pareto_tariff_df["electricity_cost"]) / baseline_tariff
pareto_tariff_df["emissions_pct"] = 100*(baseline_aef - pareto_tariff_df["emissions"]) / baseline_aef

# Plot the results  
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
fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 160 / 25.4), layout="tight")

# wholesale plot 
sns.lineplot(pareto_wholesale_df, y = "electricity_cost_pct", x = "emissions_pct", hue = "system", 
                palette=color_map, legend=False, ax=ax[0,0])
for sys in systems.keys(): 
    sys_df = pareto_wholesale_df[pareto_wholesale_df["system"] == sys]
    ax[0,0].scatter(sys_df["emissions_pct"], sys_df["electricity_cost_pct"], 
                marker=systems[sys]["marker_shape"],
                facecolor=color_map[sys],
                edgecolor='k', 
                linewidth=0.5,
                s=20,
                zorder=3
                )

ax[0,0].scatter(0, 0, color="black", marker="o", s=40, label="Baseline Wholesale")

# tariff plot 
l1 = sns.lineplot(pareto_tariff_df, y = "electricity_cost_pct", x = "emissions_pct", hue = "system", 
                palette=color_map, ax=ax[0,1])

for sys in systems.keys():
    sys_df = pareto_tariff_df[pareto_tariff_df["system"] == sys]
    ax[0,1].scatter(sys_df["emissions_pct"], sys_df["electricity_cost_pct"], 
                marker=systems[sys]["marker_shape"],
                facecolor=color_map[sys],
                edgecolor='k', 
                linewidth=0.5,
                s=20,
                zorder=3
                )
ax[0,1].scatter(0, 0, color="black", marker="o", s=40, label="Baseline Tariff")

# xaxis labels / range 
xlabel = "Scope 2 Emissions Savings (%)"
ax[0,0].set_xlabel(xlabel)
ax[0,1].set_xlabel(xlabel)
ax[0,0].set_xlim(0., 30)
ax[0,1].set_xlim(-10, 25)

# yaxis labels / range 
ylabel = "Electricity Cost Savings (%)"
ax[0,0].set_ylabel(ylabel, labelpad=8) 
ax[0,1].set_ylabel(ylabel, labelpad=1)
ax[0,0].set_ylim(0, 40)
ax[0,1].set_ylim(-150, 40)

# set xticks
ax[0,0].set_xticks(np.arange(0., 31, 5))
ax[0,1].set_xticks(np.arange(-10, 21, 5))

# # set yticks
ax[0,0].set_yticks(np.arange(0., 41, 5))
ax[0,1].set_yticks(np.arange(-150, 41, 20))

handles, _ = l1.get_legend_handles_labels()
ax[0,1].legend().remove()

# system_titles = ["Maximum Savings", "25% Uptime, 0% Power Capacity", "50% Uptime, 50% Power Capacity", "100% Uptime, 25% Power Capacity"]

for sys in systems.keys(): 
    system_name = sys
    pareto_wholesale = pd.read_csv(basepath + "/paper_figures/processed_data/pareto_wholesale/pareto_front_wholesale_{}.csv".format(system_name))
    shadow_cost, emissions = get_mac(pareto_wholesale)
    emissions_pct = 100*(baseline_mef/num_days - emissions) / (baseline_mef/num_days)
    ax[1,0].plot(emissions_pct, 
                shadow_cost, 
                label=system_name, 
                color=color_map[system_name]
                )
    ax[1,0].scatter(emissions_pct,
                shadow_cost, 
                marker=systems[sys]["marker_shape"],
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=20,
                zorder=3
                ) 

    pareto_tariff = pd.read_csv(basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name))
    shadow_cost, emissions = get_mac(pareto_tariff)
    emissions_pct = 100*(baseline_aef - emissions) / baseline_aef
    ax[1,1].plot(emissions_pct, shadow_cost, label=system_name, color=color_map[system_name])
    ax[1,1].scatter(emissions_pct,
                shadow_cost, 
                marker=systems[sys]["marker_shape"],
                facecolor=color_map[system_name],
                edgecolor='k', 
                linewidth=0.5,
                s=20,
                zorder=3
                ) 

# overlay scc line
ax[1,0].hlines(141.6, -100, 100, ls="--", color="black", label="Social Cost of Carbon")
ax[1,1].hlines(141.6, -100, 100, ls="--", color="black", label="Social Cost of Carbon")

ax[1,0].set(
    xlabel="Scope 2 Emissions Savings (%)",
    ylabel="Cost of Abatement ($/ton)",
    yscale="log",
    ylim=(0.005, 1e5),
    xlim=(0., 30),
)
ax[1,1].set(
    xlabel="Scope 2 Emissions Savings (%)",
    ylabel="Cost of Abatement ($/ton)",
    yscale="log",
    ylim=(0.008, 1e5),
    xlim=(-10, 25),
)

subplot_labels = ['a.', 'b.', 'c.', 'd.']
# add subplot labels
for a in ax.flatten():
    a.text(-0.15, 1.03, subplot_labels.pop(0), transform=a.transAxes,
           fontsize=7, fontweight='bold', va='top', ha='left')

# save figure 
for figure_type in ["png","svg", "pdf"]: 
    fig.savefig(os.path.join(basepath, "paper_figures/figures", figure_type, "pareto_emissions_cost_curve." + figure_type),
        dpi=300, bbox_inches="tight",
    )