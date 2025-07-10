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

# define overlay parameters
overlay_params = {
    'scc': {
        'face_color': 'black',
        'edge_color': 'black',
        'alpha': 0.5,
    },
    'rec': {
        'face_color': 'lightgrey',
        'edge_color': 'grey',
        'alpha': 0.5
    }
}

generate_data = False


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
    ##
    pass

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

width = 0.15

offset = [-width * 1.6, -width * 0.55, width * 0.55, width * 1.6]  # Offset for each system's shadow price for better visibility
colors= ["#FF6347", "#A9A9A9", "#FFD700", "#008080"]  # Colors for each system


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

def _add_scc_and_rec(ax, regions, width, scc=True, rec=True, plot_scc_by="mean", emission_basis="mef"):
    """
    """
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scc_df = pd.read_csv(os.path.join(basepath, "data", "offsets", "scc.csv"))
    rec_df = pd.read_csv(os.path.join(basepath, "data", "offsets", "rec.csv"))
        
    def _create_arrow_label(text, xy, xytext, rad=0.2, va='bottom'):
        ax.annotate(text, xy=xy, xytext=xytext,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, 
                                   connectionstyle=f'arc3,rad={rad}'),
                   ha='center', va=va, fontsize=18)
    
    if scc:  # Plot scc
        if plot_scc_by == "mean":  # Use 50th percentile to create "line"
            scc_bottom = scc_df[scc_df['percentile'] == 50]['value'].item()
            scc_top = scc_bottom
        else:  # Use 25th and 75th percentiles
            scc_bottom = scc_df[scc_df['percentile'] == 25]['value'].item()
            scc_top = scc_df[scc_df['percentile'] == 75]['value'].item()
        
        scc_rect = Rectangle(
            (0 - width*2.5, scc_bottom),  # x, y (left edge, bottom)
            len(regions) + width*5,  # width to cover all regions
            scc_top - scc_bottom,  # height
            facecolor=overlay_params['scc']['face_color'],
            edgecolor=overlay_params['scc']['edge_color'],
            alpha=overlay_params['scc']['alpha'],
            zorder=0,  # behind the bars
        )
        ax.add_patch(scc_rect)
        
        # Arrow pointing to SCC box
        _create_arrow_label('Social Cost\nof Carbon', 
                           (len(regions)/2 + 0.1, scc_top), 
                           (len(regions)/2 - 0.5, ax.get_ylim()[0] + 500), 
                           rad=-0.2)

    if rec:  # Add REC boxes
        # converting to float
        rec_df['price'] = rec_df['price'].astype(float)
        
        # Calculate average REC price for each ISO by type
        iso_rec_prices_by_type = {}
        national_data = rec_df[rec_df['iso'].str.lower() == 'national']
        national_averages = {}
        # For AEF basis, only use voluntary RECs
        rec_types_to_use = ['voluntary'] if emission_basis.lower() == "aef" else ['compliance', 'voluntary']
        for rec_type in rec_types_to_use:
            type_data = national_data[national_data['type'] == rec_type]
            if len(type_data) > 0:
                national_averages[rec_type] = type_data['price'].mean()
        
        for region in regions:
            region_lower = region.lower()
            region_data = rec_df[rec_df['iso'].str.lower() == region_lower]
            
            type_averages = {}

            if len(region_data) > 0:
                # Group by type (voluntary only for AEF) and calculate average REC prices for ISO
                rec_types_to_check = ['voluntary'] if emission_basis.lower() == "aef" else ['compliance', 'voluntary', 'srec']
                for rec_type in rec_types_to_check:
                    type_data = region_data[region_data['type'] == rec_type]
                    if len(type_data) > 0:
                        type_averages[rec_type] = type_data['price'].mean()
            
            # For regions with specific data, only use national averages for missing types
            if len(type_averages) > 0:
                for rec_type in rec_types_to_use:
                    if rec_type not in type_averages and rec_type in national_averages:
                        type_averages[rec_type] = national_averages[rec_type]
                iso_rec_prices_by_type[region] = type_averages
            else:
                # Use national average if no specific data for ISO
                if national_averages:
                    iso_rec_prices_by_type[region] = national_averages
        
        # Plot REC values for each ISO
        for region_idx, region in enumerate(regions):
                
            # Get hourly average emission factors using the helper function
            monthly_hourly_avg_emission_ton_per_mwh = ec.get_hourly_average_emission_factors(region, emission_basis)
            min_emission, max_emission = np.min(monthly_hourly_avg_emission_ton_per_mwh), np.max(monthly_hourly_avg_emission_ton_per_mwh)
            
            # Calculate max/min REC price equivalent for all REC types
            all_rec_prices = list(iso_rec_prices_by_type[region].values())
            min_rec_price_emission, max_rec_price_emission = min(all_rec_prices) / max_emission,  max(all_rec_prices) / min_emission
            
            # Rectangle spanning bars for ISO showing overall REC range
            rec_rect = Rectangle(
                (region_idx - width*2.5, min_rec_price_emission),  # x, y (left edge, bottom)
                width*5,  # width to cover all bars
                max_rec_price_emission - min_rec_price_emission,  # height
                facecolor=overlay_params['rec']['face_color'],
                edgecolor=overlay_params['rec']['edge_color'],
                alpha=0.5,
                zorder=0  # behind the bars
            )
            ax.add_patch(rec_rect)
            
            # Arrow pointing to ERCOT REC box
            if region == "ERCOT":
                ax.annotate('Typical\nREC Price\nRange', 
                           xy=(region_idx, min_rec_price_emission), 
                           xytext=(region_idx - 0.5, ax.get_ylim()[0] + 2),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5, 
                                           connectionstyle='arc3,rad=0.3'),
                           ha='center', va='top', fontsize=18)
    
# Comment this line out out to disable SCC and REC overlays
_add_scc_and_rec(
    ax, regions=regions, width=width, scc=True, rec=True, plot_scc_by="mean", emission_basis="mef"
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
