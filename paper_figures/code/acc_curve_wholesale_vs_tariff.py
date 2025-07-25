import calendar, os, json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import analysis.maxsavings as ms 
from analysis.pricesignal import getmef, getaef, getdam, gettariff
from analysis.acc_curve import acc_curve
from electric_emission_cost import costs
from electric_emission_cost.units import u
from electric_emission_cost import utils

# import color maps as json
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

# VB parameter settings 
systems = {
    "maxflex" : {
        "system_uptime": 0.0,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "uptime_equality": False,
        "pareto_stepsize": 0.01
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
        "pareto_stepsize": 0.05 
    },
}

# grid parameter settings 
region = "CAISO"
month = 7
year = 2023

# data/figure gen settings 
generate_data = True
threads = 60 

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
        baseload = np.ones_like(mef)

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
        baseload = np.ones_like(aef)

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

else: 
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
baseline_dam = getdam(region, month, basepath).sum() * num_days / 1000      # convert $/MWh to $/kWh
baseline_mef = getmef(region, month, basepath).sum() * num_days / 1000**2   # convert kg/MWh to mton/kWh
baseline_aef = getaef(region, month, basepath).sum() * num_days / 1000**2   # convert kg/MWh to mton/kWh

# calculate tariff baseline
tariff = gettariff(region, basepath=basepath, full_list=False) 
baseload = np.ones(num_days*4*24)
charge_dict = costs.get_charge_dict(startdate_dt, enddate_dt, tariff, resolution="15m")
baseline_tariff, _= costs.calculate_cost(
    charge_dict,
    {"electric":baseload,
    "gas": np.zeros_like(baseload)},
    resolution="15m",
    consumption_estimate=0,
)

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
fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 80 / 25.4), layout="tight")

# wholesale plot 
sns.lineplot(pareto_wholesale_df, y = "electricity_cost", x = "emissions", hue = "system", 
                palette=color_map, legend=False, ax=ax[0])
ax[0].scatter(baseline_mef, baseline_dam, color="black", marker="o", s=50, label="Baseline Wholesale")

# tariff plot 
l1 = sns.lineplot(pareto_tariff_df, y = "electricity_cost", x = "emissions", hue = "system", 
                palette=color_map, ax=ax[1])
ax[1].scatter(baseline_aef, baseline_tariff, color="black", marker="o", s=50, label="Baseline Tariff")

# xaxis labels / range 
xlabel = "Emissions (tons CO$_2$)"
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_xlim(0.20, 0.30)
ax[1].set_xlim(0.10, 0.20)

# yaxis labels / range 
ylabel = "Electricity Cost ($)"
ax[0].set_ylabel(ylabel, labelpad=1) 
ax[1].set_ylabel(ylabel, labelpad=1)
ax[0].set_ylim(-20, 50)
ax[1].set_ylim(0, 600)

# set xticks
ax[0].set_xticks(np.arange(0.20, 0.31, 0.02))
ax[1].set_xticks(np.arange(0.10, 0.21, 0.02))

# set yticks
ax[0].set_yticks(np.arange(-20, 51, 10))
ax[1].set_yticks(np.arange(0, 601, 100))

handles, _ = l1.get_legend_handles_labels()
ax[1].legend().remove()

system_titles = ["Maximum Savings", "25% Uptime, 0% Power Capacity", "50% Uptime, 50% Power Capacity", "100% Uptime, 25% Power Capacity"]

subplot_labels = ['a.', 'b.']
# add subplot labels
for a in ax.flatten():
    a.text(-0.15, 1.03, subplot_labels.pop(0), transform=a.transAxes,
           fontsize=7, fontweight='bold', va='top', ha='left')

# save figure 
for figure_type in ["png","svg", "pdf"]: 
    fig.savefig(os.path.join(basepath, "paper_figures/figures", figure_type, "pareto_emissions_cost_curve." + figure_type),
        dpi=300, bbox_inches="tight",
    )