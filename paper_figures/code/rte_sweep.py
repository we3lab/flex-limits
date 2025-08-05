# imports
import os, json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.pricesignal import getmef, getdam, getaef, gettariff
from analysis.savings import (
    mef_savings, 
    aef_savings, 
    dam_savings, 
    tariff_savings
)
from analysis.maxsavings import (
    get_start_end
)

GENERATE_DATA = False

region = "CAISO"
month = 7

start_date_dt, end_date_dt = get_start_end(month)

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

# VB parameter settings 
systems = {
    "maxflex" : {
        "system_uptime": 0.0,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "uptime_equality": False,
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0, 
        "uptime_equality": True, 
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "uptime_equality": True, 
    },
    "100uptime_25flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.25, 
        "uptime_equality": True, 
    }
}

# import colors 
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors = colors["examplesys_colors"]
incentive_colors = colors["incentive_colors"]


if GENERATE_DATA:
    rtes = np.linspace(0.6, 1.0, 40, endpoint=True)  # round-trip efficiencies from 60% to 100%

    # get data
    mef = getmef(region, month)
    dam = getdam(region, month)
    aef = getaef(region, month)
    tariff = gettariff(region)
    start, end = get_start_end(month)

    mef_pct_savings = np.zeros((len(systems),len(rtes)))
    dam_pct_savings = np.zeros((len(systems),len(rtes)))
    aef_pct_savings = np.zeros((len(systems),len(rtes)))
    tariff_pct_savings = np.zeros((len(systems), len(rtes)))

    savings_df = pd.DataFrame(columns=["region", "month", "incentive", "sys", "rte", "savings"])

    for idx, r in enumerate(rtes):
        for s, sys in enumerate(systems.keys()):
            pct_mef = mef_savings(
                data=mef,
                rte=r,
                system_uptime=systems[sys]["system_uptime"],
                continuous_flex=systems[sys]["continuous_flexibility"],
                baseload=np.ones_like(mef)
            )
            mef_pct_savings[s, idx] = pct_mef
            savings_df = pd.concat([savings_df, pd.DataFrame({
                "region": region,
                "month": month,
                "incentive": "mef",
                "sys": sys,
                "rte": r,
                "savings": pct_mef
                }, index=[0])], ignore_index=True)

            pct_dam = dam_savings(
                data=dam,
                rte=r,
                system_uptime=systems[sys]["system_uptime"],
                continuous_flex=systems[sys]["continuous_flexibility"],
                baseload=np.ones_like(dam)
            )
            dam_pct_savings[s, idx] = pct_dam
            savings_df = pd.concat([savings_df, pd.DataFrame({
                "region": region,
                "month": month,
                "incentive": "dam",
                "sys": sys,
                "rte": r,
                "savings": pct_dam
                }, index=[0])], ignore_index=True)

            pct_aef = aef_savings(
                data=aef,
                rte=r,
                system_uptime=systems[sys]["system_uptime"],
                continuous_flex=systems[sys]["continuous_flexibility"],
                baseload=np.ones_like(aef)
            )
            aef_pct_savings[s, idx] = pct_aef
            savings_df = pd.concat([savings_df, pd.DataFrame({
                "region": region,
                "month": month,
                "incentive": "aef",
                "sys": sys,
                "rte": r,
                "savings": pct_aef
                }, index=[0])], ignore_index=True)

            pct_tariff = tariff_savings(
                data=tariff,
                rte=r,
                system_uptime=systems[sys]["system_uptime"],
                continuous_flex=systems[sys]["continuous_flexibility"],
                baseload=np.ones(744),
                startdate_dt=start_date_dt,
                enddate_dt=end_date_dt
            )
            tariff_pct_savings[s, idx] = pct_tariff
            savings_df = pd.concat([savings_df, pd.DataFrame({
                "region": region,
                "month": month,
                "incentive": "tariff",
                "sys": sys,
                "rte": r,
                "savings": pct_tariff
                }, index=[0])], ignore_index=True)


    # save data
    savings_df.to_csv(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "savings_rte_sweep.csv"), 
        index=False
    )

# read data from csv
savings_df = pd.read_csv(
    os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data", "savings_rte_sweep.csv"))
)

systems_order = list(systems.keys())
mef_pct_savings = savings_df[savings_df["incentive"] == "mef"].pivot(
    index="sys",
    columns="rte",
    values="savings"
)
mef_pct_savings = mef_pct_savings.reindex(systems_order).values

dam_pct_savings = savings_df[savings_df["incentive"] == "dam"].pivot(
    index="sys",
    columns="rte",
    values="savings"
)
dam_pct_savings = dam_pct_savings.reindex(systems_order).values

aef_pct_savings = savings_df[savings_df["incentive"] == "aef"].pivot(
    index="sys",
    columns="rte",
    values="savings"
)
aef_pct_savings = aef_pct_savings.reindex(systems_order).values

tariff_pct_savings = savings_df[savings_df["incentive"] == "tariff"].pivot(
    index="sys",
    columns="rte",
    values="savings"
)
tariff_pct_savings = tariff_pct_savings.reindex(systems_order).values

# plot for MEF savings 
fig, ax = plt.subplots(2,2, figsize=(180/25.4, 150/25.4))
    
for s, sys in enumerate(systems.keys()):
    ax[0,0].plot(rtes*100, mef_pct_savings[s,:], label=sys, color=sys_colors[sys], linewidth=2)
    ax[0,1].plot(rtes*100, dam_pct_savings[s,:], label=sys, color=sys_colors[sys], linewidth=2)
    ax[1,0].plot(rtes*100, aef_pct_savings[s,:], label=sys, color=sys_colors[sys], linewidth=2)
    ax[1,1].plot(rtes*100, tariff_pct_savings[s,:], label=sys, color=sys_colors[sys], linewidth=2, alpha=0.5)


ax[0,0].set_title("Marginal Emissions Factor")
ax[0,1].set_title("Day-Ahead Market")
ax[1,0].set_title("Average Emissions Factor")
ax[1,1].set_title("Tariff")

for a in ax.flatten():
    a.hlines(0, 60, 100, color='k', lw=1, ls="--")
    a.set(
        xlabel="Round-trip Efficiency (%)",
        ylabel="Percentage Savings (%)",
        xlim=(60, 100),
        ylim=(-50, 50),
        xticks=np.arange(60, 101, 5),
        yticks=np.arange(-50, 51, 10),
    )

fig.tight_layout()