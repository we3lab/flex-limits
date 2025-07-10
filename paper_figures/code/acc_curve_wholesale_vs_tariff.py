import calendar
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import analysis.maxsavings as ms 
from analysis.pricesignal import getmef, getaef, getdam, gettariff
from analysis.acc_curve import acc_curve




systems = {
    "maxflex" : {
        "system_uptime": 1/24,  # minimum uptime
        "continuous_flexibility": 0.99, # full flexibility
        "pareto_stepsize": 0.05
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0, 
        "pareto_stepsize": 0.1  
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "pareto_stepsize": 0.1 
    },
    "100uptime_75flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.75, 
        "pareto_stepsize": 0.1 
    },
}

region = "CAISO"
month = 4
year = 2023

generate_data = False  
basepath =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#get wholesale energy market + marginal emissions 
dam = getdam(region, month, basepath)
mef = getmef(region, month, basepath)

#get tariff + average emissions 
_, num_days = calendar.monthrange(year, month = month) 
startdate_dt, enddate_dt = ms.get_start_end(month)
tariff = gettariff(region, basepath=basepath)
aef = getaef(region, month, basepath)   
aef = np.tile(aef, num_days) 


if generate_data: 

    pareto_wholesale_list = []
    pareto_tariff_list = []
    for system_name, system_dict in systems.items(): 

        
        system_uptime = systems[system_name]["system_uptime"]
        continuous_flexibility = systems[system_name]["continuous_flexibility"]
        pareto_stepsize = systems[system_name]["pareto_stepsize"]

        # dam and mef pareto front 
        baseload = np.ones_like(mef)

        battery_params = {
            "baseload":baseload, 
            "emissions_signal": mef, 
            "emissions_type": "mef", 
            "cost_signal": dam, 
            "cost_type": "dam",
            # "cost_signal_units": u.USD / u.MWh,
            "flex_capacity": continuous_flexibility, 
            "min_onsteps": max(int(len(mef) * (system_uptime)), 1), 
            "startdate_dt": None, 
            "enddate_dt": None
        }

        acc = acc_curve(battery_params=battery_params)
        em_opt = acc.calc_emissions_optimal()
        cost_opt = acc.calc_cost_optimal()
        savepath = basepath + "/paper_figures/processed_data/pareto_wholesale/pareto_front_wholesale_{}.csv".format(system_name)
        pareto_sys_df = acc.build_pareto_front(stepsize = pareto_stepsize, savepath = savepath)
        pareto_sys_df["system"] = system_name 
        pareto_wholesale_list.append(pareto_sys_df)


        # tariff and aef pareto front 
        baseload = np.ones_like(aef)

        battery_params = {
            "baseload":baseload, 
            "emissions_signal": aef, 
            "emissions_type": "aef", 
            "cost_signal": tariff, 
            "cost_type": "tariff",
            # "cost_signal_units": u.USD / u.kWh, 
            "flex_capacity": continuous_flexibility, 
            "min_onsteps": max(int(len(aef) * (system_uptime)), 1), 
            "startdate_dt": startdate_dt, 
            "enddate_dt": enddate_dt
        }

        acc = acc_curve(battery_params=battery_params)
        em_opt = acc.calc_emissions_optimal()
        cost_opt = acc.calc_cost_optimal()
        savepath = basepath + "/paper_figures/processed_data/pareto_tariff/pareto_front_tariff_{}.csv".format(system_name)
        pareto_sys_df = acc.build_pareto_front(stepsize = pareto_stepsize, savepath = savepath)
        pareto_sys_df["system"] = system_name 
        pareto_tariff_list.append(pareto_sys_df)

else: 

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


pareto_wholesale_df = pd.concat(pareto_wholesale_list).reset_index(drop = True)
pareto_tariff_df = pd.concat(pareto_tariff_list).reset_index(drop = True)

# Calculate wholesale emissions/costs for entire month 
pareto_wholesale_df["emissions"] *= num_days
pareto_wholesale_df["electricity_cost"] *=num_days

system_titles = ['Maximum Savings', '25% Uptime, 0% Power Capacity', '50% Uptime, 50% Power Capacity', '100% Uptime, 25% Power Capacity']

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Colors for each system
num_systems = len(systems)  # Number of systems to plot
system_names = list(systems.keys())
color_map = dict(zip(system_names, colors))


fig, ax = plt.subplots(1, 2, figsize = (6.5, 3.5))

# wholesale plot 
sns.lineplot(pareto_wholesale_df, x = "electricity_cost", y = "emissions", hue = "system", 
                palette=color_map, legend=False, ax=ax[0])

# tariff plot 
l1 = sns.lineplot(pareto_tariff_df, x = "electricity_cost", y = "emissions", hue = "system", 
                palette=color_map, ax=ax[1])


# yaxis labels / range 
ylabel = "Emissions (tons $CO_2$)"
ax[0].set_ylabel(ylabel)
ax[1].set_ylabel(ylabel)
ax[0].set_ylim(0.15, 0.30)
ax[1].set_ylim(0.1, 0.20)

# xaxis labels / range 
xlabel = "Electricity Cost ($)"
ax[0].set_xlabel(xlabel) 
ax[1].set_xlabel(xlabel)
ax[0].set_xlim(-15, 30)
ax[1].set_xlim(0, 500)


handles, _ = l1.get_legend_handles_labels()
ax[1].legend().remove()


fig.legend(handles, system_titles, ncol = 2, loc='lower left', bbox_to_anchor=(0.05, -0.15), frameon = False)

plt.tight_layout()  


fig.savefig(os.path.join(basepath, "paper_figures/figures/png", "pareto_emissions_cost_curve.png"),
    dpi=300, bbox_inches="tight",
)

fig.savefig(
    os.path.join(basepath, "paper_figures/figures/pdf", "pareto_emissions_cost_curve.pdf"),
    dpi=300, bbox_inches="tight",
)





