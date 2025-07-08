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
        'line_color': 'black',
        'line_alpha': 0.7,
        'line_width': 2,
        'box_alpha': 0.3,
        'box_color': 'black'
    },
    'rec': {
        'face_color': 'white',
        'edge_color': 'black',
        'alpha': 0.5,
        'hatching_intensity': 5,
        'zorder': 0
    }
}

generate_data = False


regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]

systems = {
    "maxflex" : {
        "system_uptime": 1/24,  # minimum uptime
        "continuous_flexibility": 0.0
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

def _add_scc_and_rec(ax, regions, ps, width, scc=True, rec=True, plot_scc_by="mean"):
    scc_df = pd.read_csv("data/offsets/scc.csv")
    rec_df = pd.read_csv("data/offsets/rec.csv")
    
    legend_info = {'handles': [], 'labels': []}
    
    if scc:  # Add SCC
        if plot_scc_by == "mean":  # Plot as line 
            scc_price_usd_ton = scc_df[scc_df['value_type'] == 'mean']['value'].item()
            scc_line = ax.axhline(
                y=scc_price_usd_ton, 
                color=overlay_params['scc']['line_color'], 
                alpha=overlay_params['scc']['line_alpha'], 
                linewidth=overlay_params['scc']['line_width'],
                zorder=overlay_params['rec']['zorder']
            )
            legend_info['handles'].append(scc_line)
            legend_info['labels'].append('Social Cost of Carbon')
        elif plot_scc_by == "max_min":  # Plot as box
            scc_min = scc_df[scc_df['value_type'] == 'min']['value'].item()
            scc_max = scc_df[scc_df['value_type'] == 'max']['value'].item()
            scc_rect = Rectangle(
                (0 - width*2.5, scc_min),  # x, y (left edge, bottom)
                len(regions) + width*5,  # width to cover all regions
                scc_max - scc_min,  # height
                facecolor=overlay_params['scc']['box_color'],
                edgecolor=overlay_params['scc']['box_color'],
                alpha=overlay_params['scc']['box_alpha'],
                zorder=overlay_params['rec']['zorder'],  # behind the bars
            )
            ax.add_patch(scc_rect)
            scc_patch = Patch(
                facecolor=overlay_params['scc']['box_color'], 
                alpha=overlay_params['scc']['box_alpha'], 
                edgecolor=overlay_params['scc']['box_color'], 
            )
            legend_info['handles'].append(scc_patch)
            legend_info['labels'].append('Social Cost of Carbon')

    if rec:  # Add REC boxes
        # converting to float
        rec_df['price'] = rec_df['price'].astype(float)
        
        rec_params = {
            'compliance': {
                'hatch': '\\'*overlay_params['rec']['hatching_intensity'],
                'label': 'Compliance REC'
            },
            'voluntary': {
                'hatch': '.'*overlay_params['rec']['hatching_intensity'],
                'label': 'Voluntary REC'
            },
            'srec': {
                'hatch': '+'*overlay_params['rec']['hatching_intensity'],
                'label': 'SREC'
            }
        }
        
        # Calculate average REC price for each ISO by type
        iso_rec_prices_by_type = {}
        national_data = rec_df[rec_df['iso'].str.lower() == 'national']
        national_averages = {}
        if len(national_data) > 0:
            for rec_type in ['compliance', 'voluntary']:  # national has compliance and voluntary
                type_data = national_data[national_data['type'] == rec_type]
                if len(type_data) > 0:
                    national_averages[rec_type] = type_data['price'].mean()
        
        for region in regions:
            region_lower = region.lower()
            region_data = rec_df[rec_df['iso'].str.lower() == region_lower]
            
            type_averages = {}
            
            if len(region_data) > 0:
                # Group by type and calculate average REC prices for ISO
                for rec_type in ['compliance', 'voluntary', 'srec']:
                    type_data = region_data[region_data['type'] == rec_type]
                    if len(type_data) > 0:
                        type_averages[rec_type] = type_data['price'].mean()
            
            # For regions with specific data, only use national averages for missing types
            if len(type_averages) > 0:
                for rec_type in ['compliance', 'voluntary']:
                    if rec_type not in type_averages and rec_type in national_averages:
                        type_averages[rec_type] = national_averages[rec_type]
                
                iso_rec_prices_by_type[region] = type_averages
            else:
                # Use national average if no specific data for ISO
                if national_averages:
                    iso_rec_prices_by_type[region] = national_averages
        
        # Plot REC values for each ISO by type
        for region_idx, region in enumerate(regions):
            # Gather MEF values (kg/MWh) for this ISO and calculate hourly averages for each month
            monthly_hourly_avg_mef_kg_per_mwh = []
            for month in range(1, 13):
                mef_data = ps.getmef(region, month)  # for 24 hr
                # Calculate hourly average for this month
                hourly_avg_mef = np.mean(mef_data)
                monthly_hourly_avg_mef_kg_per_mwh.append(hourly_avg_mef)
            
            # Calculate REC $/kg equivalent using max and min of hourly-averaged MEF
            monthly_hourly_avg_mef_ton_per_mwh = np.array(monthly_hourly_avg_mef_kg_per_mwh) / 1000  # kg to ton
            
            # Plot each REC type for this region
            for rec_type, rec_price_usd_mw in iso_rec_prices_by_type[region].items():
                # REC price in $/MWh = $/MW * 1 hr
                # REC value in $/ton = ($/MWh) / (ton/MWh) = $/ton
                rec_price_min_mef = rec_price_usd_mw / np.max(monthly_hourly_avg_mef_ton_per_mwh)
                rec_price_max_mef = rec_price_usd_mw / np.min(monthly_hourly_avg_mef_ton_per_mwh)
                
                # Rectangle spanning bars for ISO showing REC range with type-specific hatching
                rec_rect = Rectangle(
                    (region_idx - width*2.5, rec_price_min_mef),  # x, y (left edge, bottom)
                    width*5,  # width to cover all bars
                    rec_price_max_mef - rec_price_min_mef,  # height
                    facecolor=overlay_params['rec']['face_color'],
                    edgecolor=overlay_params['rec']['edge_color'],
                    alpha=overlay_params['rec']['alpha'],
                    hatch=rec_params[rec_type]['hatch'],
                    zorder=overlay_params['rec']['zorder'],  # behind the bars
                    label=None
                )
                ax.add_patch(rec_rect)
        
        # Add legend entries for each REC type
        for rec_type in ['compliance', 'voluntary', 'srec']:
            if rec_type in rec_params:
                rec_patch = Patch(
                    facecolor=overlay_params['rec']['face_color'], 
                    edgecolor=overlay_params['rec']['edge_color'], 
                    alpha=overlay_params['rec']['alpha'],
                    hatch=rec_params[rec_type]['hatch'],
                    label=rec_params[rec_type]['label']
                )
                legend_info['handles'].append(rec_patch)
                legend_info['labels'].append(rec_params[rec_type]['label'])
    
    ax.legend(legend_info['handles'], legend_info['labels'], loc='upper left', frameon=False, fontsize=14)

# Comment this line out out to disable SCC and REC overlays
_add_scc_and_rec(
    ax, regions=regions, ps=ps, width=width, scc=True, rec=True, plot_scc_by="mean"
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
