import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from analysis import pricesignal as ps
from analysis import emissionscost as ec


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
    "100uptime_75flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.75, 
        "uptime_equality": True
    },
}

# grid parameter settings 
months = [1,7]
regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]

# data/figure gen settings 
generate_data = False
threads = 20 
figure_type = "svg"

paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if generate_data == True:

    # import tariff data 
    tariff_costing_path = os.path.join(os.path.dirname(paperfigs_basepath),"data","tariff_wwtp", "WWTP_Billing.xlsx")
    tariff_metadata_path = os.path.join(os.path.dirname(paperfigs_basepath),"data","tariff_wwtp", "metadata_iso.csv")
    tariffs_df = pd.read_csv(tariff_metadata_path)

    for region in regions:
        # create a new dataframe for all tariffs in the specified region 
        region_tariffs, region_tariff_ids = ps.gettariff(region=region, full_list=True, return_ids=True)
        print("Region: {}, Number of Tariffs {}:".format(region, len(region_tariffs)))
       
        for system_name, params in systems.items():
            results_dicts = []
            t0 = time.time()
            for n, tariff in enumerate(region_tariffs): 
        
                for month in months:
                    try: 
                        # calculate shadow cost for the given case (system, tariff, month)
                        tmp = ec.shadowcost_tariff(region=region,
                                                            tariff_data=tariff, 
                                                            month=month,
                                                            system_uptime=params["system_uptime"],
                                                            continuous_flexibility=params["continuous_flexibility"], 
                                                            uptime_equality=params["uptime_equality"], 
                                                            threads = threads
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
                    t1 = time.time()

            print("time for {} system {}: {}".format(region, system_name, t1 - t0))
                

            # collapse the list of dicts into a dict of lists []
            sys_results_df = pd.DataFrame(results_dicts)

            # rearrange the columns
            cols_first = ["region","system","month", "tariff_id"]
            cols = cols_first + [col for col in sys_results_df.columns if col not in cols_first]  
            sys_results_df = sys_results_df[cols]  

            # save df in results folder as a csv file
            sys_results_df.to_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_tariff", f"{region}_{system_name}.csv"), index=False)

            del sys_results_df, results_dicts
        
        del region_tariffs

else:
    pass

# read in results
results_list = []
sys_list = []
region_list = []
for region_idx, region in enumerate(regions):
    # Iterate over each system and plot the results
    for sys in systems.keys():
        # Load the results for the current region and system 
        sys_results_df = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_tariff", f"{region}_{sys}.csv"))
        results_list.append(sys_results_df)

# create combined data frame                             
results_df = pd.concat(results_list, ignore_index=True)
results_df.dropna(how = "any", inplace=True)
results_df.loc[results_df.shadow_price_usd_ton < 1e-3, "shadow_price_usd_ton"] = 1e-3 #clip very small/negative prices (necessary for log scale)


# Plot the results  
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
        "legend.fontsize": 22,
        "ytick.labelsize": 22, 
    }
)

# create color map 
colors = ["#FF6347", "#A9A9A9", "#FFD700", "#008080"] # Colors for each system
num_systems = len(systems)  # Number of systems to plot
system_names = list(systems.keys())
color_map = dict(zip(system_names, colors))

# create figure 
fig, ax = plt.subplots(figsize=(12, 8))

# violin plots 
p1 = sns.violinplot(data=results_df, x = "region", y ="shadow_price_usd_ton", hue = "system", 
                    gap = 0.4, inner = "point", density_norm="width", 
                    inner_kws= {"s": 15, "alpha":0.5,}, alpha = 0.9, 
                   palette=color_map, log_scale=True, split=True,  ax=ax)

ax.set(
    xlabel="Region",
    xticks=np.arange(len(regions)),
    xticklabels=regions,
    ylabel="Cost of Abatement (USD/ton $CO_2$)",
    ylim=(1e-2, 1e5),
    yscale="log"
)


# create legend 
handles, _ = p1.get_legend_handles_labels()
system_titles = ["Maximum Savings", "25% Uptime, 0% Power Capacity", "50% Uptime, 50% Power Capacity", "100% Uptime, 25% Power Capacity"]
ax.legend(handles, system_titles, ncol = 1, loc = (0.15, -0.48), frameon = False)


# save figure 
if figure_type in ["png","svg", "pdf"]: 
    fig.savefig(os.path.join(paperfigs_basepath, "figures", figure_type, "shadowcost_tariff_violinplot." + figure_type),
        dpi=300,
        bbox_inches="tight",
    )

else: 
    raise Warning("Figure type not supported.")