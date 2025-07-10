import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

generate_data = True
months = [1,7]


regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]


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
    "100uptime_75flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.75 
    },
}


paperfigs_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tariff_costing_path = os.path.join(os.path.dirname(paperfigs_basepath),"data","tariff_wwtp", "WWTP_Billing.xlsx")
tariff_metadata_path = os.path.join(os.path.dirname(paperfigs_basepath),"data","tariff_wwtp", "metadata_iso.csv")

tariffs_df = pd.read_csv(tariff_metadata_path)


if generate_data == True:
    for region in regions:
        # create a new dataframe for all tariffs in the specified region 
        region_tariffs = tariffs_df.loc[tariffs_df.ISO == region, :]

        print('region: {}/n'.format(region))
        print(region_tariffs)
        
        for system_name, params in systems.items():
            results_dicts = []
            month_list = []
            for tariff_no in region_tariffs.CWNS_No: 
                
                for month in months:
                    tmp = ec.shadowcost_tariff(region=region,
                                                        electricity_costing_CWNS_No = str(tariff_no), 
                                                        tariff_costing_path = tariff_costing_path, 
                                                        month=month,
                                                        system_uptime=params['system_uptime'],
                                                        continuous_flexibility=params['continuous_flexibility'])

                    results_dicts.append(tmp)
                    month_list.append(month)
                
            # collapse the list of dicts into a dict of lists 
            results_df = pd.DataFrame(results_dicts)

            # add a months columns and rearrange the columns to have it first
            results_df["month"] = month_list # add a col for month
            cols = ["month"] + [col for col in results_df.columns if col != "month"]  
            results_df = results_df[cols]  

            # put the df in the params dict
            params["results"] = results_df

            # save df in results folder as a csv file
            results_df.to_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_tariff", f"{region}_{system_name}.csv"), index=False)
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
        results_df = pd.read_csv(os.path.join(paperfigs_basepath, "processed_data", "shadowcost_tariff", f"{region}_{sys}.csv"))
        
        # Plot the shadow price as a bar plot
        bottom = results_df["shadow_price_usd_ton"].min()
        height = results_df["shadow_price_usd_ton"].max() - bottom

        ax.bar(region_idx + offset[idx], 
                    height = height, 
                    bottom=bottom, 
                    width=width, 
                    label=system_name, 
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

fig.savefig(os.path.join(paperfigs_basepath, "figures/png", "shadowcost_tariff_boxplot.png"),
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    os.path.join(paperfigs_basepath, "figures/pdf", "shadowcost_tariff_boxplot.pdf"),
    dpi=300,
    bbox_inches="tight",
)