from analysis.shadowcost import shadow_cost
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os, json

GENERATE_DATA = True

# import color maps as json
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)

cmaps=colors["incentive_cmaps"]
sys_colors=colors["examplesys_colors"]

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

def run_case(uptime, flex, region="CAISO", month=7, abatement_frac=1.0, emissions_type="mef", costing_type="dam"):
        cost=shadow_cost(system_uptime=uptime,
                        power_capacity=flex,
                        region=region,
                        month=month,
                        uptime_equality=True,
                        abatement_frac=abatement_frac,
                        baseload=None,
                        emissions_type=emissions_type,
                        costing_type=costing_type
            )
        df = pd.DataFrame({
            "system_uptime": [uptime],
            "continuous_flex": [flex],
            "shadow_price_usd_ton": [cost]
        })
        return df


# def generate_data(region, month, abatement_frac=1.0, emissions_type="mef", costing_type='dam'):
#     uptimes = np.arange(0, 25, 1) / 24  # 1 to 24 hours to percent in intervals
#     continuous_flex = np.arange(0, 1.01, 0.1)  # 0 to 100%

#     df = pd.DataFrame(columns=["system_uptime", "continuous_flex", "shadow_price_usd_ton"])
#     # calculate shadow cost
#     for i, uptime in enumerate(uptimes):
#         for j, flex in enumerate(continuous_flex):
#             cost=shadow_cost(system_uptime=uptime,
#                             power_capacity=flex,
#                             region=region,
#                             month=month,
#                             uptime_equality=True,
#                             abatement_frac=abatement_frac,
#                             baseload=None,
#                             emissions_type=emissions_type,
#                             costing_type=costing_type
#                 )
#             df = pd.concat([df, pd.DataFrame({
#                 "system_uptime": [uptime],
#                 "continuous_flex": [flex],
#                 "shadow_price_usd_ton": [cost]
#             })], ignore_index=True)

#     # save data
#     savepath = os.path.join(os.path.dirname(__file__), f"processed_data/shadowcost_{costing_type}_{emissions_type}_{region}_month{month}.csv")
#     # check if directory exists, if not create it
#     os.makedirs(os.path.dirname(savepath), exist_ok=True)
#     return df


# regions = ["PJM"]
# months = [7]  # representative months
regions = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
months = np.arange(1, 13)

emissions_type = "mef"  # "mef" or "aef"
costing_type = "dam"    # "dam" or "tariff"

for region in regions:
    for month in months:
        if GENERATE_DATA == True:
            tasks = []
            uptimes = np.arange(0, 25, 1) / 24  # 1 to 24 hours to percent in intervals
            continuous_flex = np.arange(0, 1.01, 0.1)  # 0 to 100%
            for i, uptime in enumerate(uptimes):
                for j, flex in enumerate(continuous_flex):
                    tasks.append(delayed(run_case)(uptime, flex, region=region, month=month, abatement_frac=1.0, emissions_type=emissions_type, costing_type=costing_type))
            
            df = pd.concat(Parallel(n_jobs=20, backend="loky")(tasks)).reset_index(drop=True)
            savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"processed_data/shadowcost_{costing_type}_{emissions_type}_{region}_month{month}.csv")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            df.to_csv(savepath, index=False)

        # read data 
        read_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"processed_data/shadowcost_{costing_type}_{emissions_type}_{region}_month{month}.csv")
        df = pd.read_csv(read_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # df = df[df["shadow_price_usd_ton"] > -1e-2]  # remove negatives
        df.loc[df["shadow_price_usd_ton"] < -1e-2, "shadow_price_usd_ton"] = np.nan 
        df.loc[df["shadow_price_usd_ton"] < 0.01, "shadow_price_usd_ton"] = 0.01 # replace values < 0.01 with 0.01 for log scale


        # pivot data for contour plot
        pivot_df = df.pivot(index="continuous_flex", columns="system_uptime", values="shadow_price_usd_ton")
        X, Y = np.meshgrid(pivot_df.index, pivot_df.columns)
        Z = pivot_df.values.T

        # plot heatmap using seaborn
        fig, ax = plt.subplots(figsize=(3.25, 2.5))
        cmap = sns.color_palette("GnBu", as_cmap=True)
        cmap.set_bad(color='lightgrey')
        heatmap = ax.contourf(X*100, Y*100, Z, cmap=cmap, vmin=1e-2, vmax=1e5, norm=LogNorm(), levels=np.logspace(-2, 5, 50))
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_ticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
        cbar.set_label('Shadow Price ($/ton CO2)', rotation=90, labelpad=5)
        ax.set_xlabel('Power Capacity [%]')
        ax.set_ylabel('System Uptime [%]')
        ax.set_title(f'{region} - month {month}')
        ax.set_yticks(np.arange(0, 101, 20))



        # save figure to folder
        for fmt in ['png', 'svg', 'pdf']:
            savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"figures/{fmt}/shadowcost_contour_{emissions_type}_{costing_type}/shadowcost_contour_{region}_month{month}.{fmt}")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath, dpi=300, bbox_inches='tight')

        plt.close()

