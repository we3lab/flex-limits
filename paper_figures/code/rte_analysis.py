# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from analysis.rte_analysis import savings_rte

# %%
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

# %%
region="CAISO"
month=7
def solve_savings_constant_uptime_dam(power_capacity, rte):
    savings= savings_rte(uptime=0.8,
                       power_capacity=power_capacity,
                       rte=rte,
                       region=region,
                       month=month,
                       signal="cost")
    return pd.DataFrame({"power_capacity": [power_capacity],
                            "rte": [rte],
                             "savings": [savings]})

def solve_savings_constant_power_dam(uptime, rte):
    savings= savings_rte(uptime=uptime,
                       power_capacity=0.5,
                       rte=rte,
                       region=region,
                       month=month,
                       signal="cost")
    return pd.DataFrame({"uptime": [uptime],
                            "rte": [rte],
                             "savings": [savings]})

def solve_savings_constant_power_mef(uptime, rte):
    savings= savings_rte(uptime=uptime,
                       power_capacity=0.5,
                       rte=rte,
                       region=region,
                       month=month,
                       signal="emissions")
    return pd.DataFrame({"uptime": [uptime],
                            "rte": [rte],
                             "savings": [savings]})

def solve_savings_constant_uptime_mef(power_capacity, rte):
    savings= savings_rte(uptime=0.8,
                       power_capacity=power_capacity,
                       rte=rte,
                       region=region,
                       month=month,
                       signal="emissions")
    return pd.DataFrame({"power_capacity": [power_capacity],
                            "rte": [rte],
                             "savings": [savings]})

# %%
tasks = []
for rte in np.linspace(0.6, 1.0, 20):
    for power_capacity in np.linspace(0.1, 1.0, 20):
        tasks.append((power_capacity, rte))
results_constant_uptime_dam = pd.concat(Parallel(n_jobs=20)(delayed(solve_savings_constant_uptime_dam)(power_capacity, rte) for power_capacity, rte in tasks), ignore_index=True)

tasks = []
for rte in np.linspace(0.6, 1.0, 20):
    for uptime in np.linspace(0.1, 1.0, 20):
        tasks.append((uptime, rte))
results_constant_power_dam = pd.concat(Parallel(n_jobs=20)(delayed(solve_savings_constant_power_dam)(uptime, rte) for uptime, rte in tasks), ignore_index=True)

tasks = []
for rte in np.linspace(0.6, 1.0, 20):
    for uptime in np.linspace(0.1, 1.0, 20):
        tasks.append((uptime, rte))
results_constant_power_mef = pd.concat(Parallel(n_jobs=20)(delayed(solve_savings_constant_power_mef)(uptime, rte) for uptime, rte in tasks), ignore_index=True)

tasks = []
for rte in np.linspace(0.6, 1.0, 20):
    for power_capacity in np.linspace(0.1, 1.0, 20):
        tasks.append((power_capacity, rte))
results_constant_uptime_mef = pd.concat(Parallel(n_jobs=20)(delayed(solve_savings_constant_uptime_mef)(power_capacity, rte) for power_capacity, rte in tasks), ignore_index=True)


# %%
fig, ax = plt.subplots(2,2, figsize=(180 / 25.4, 100 / 25.4))
pivot_table = results_constant_uptime_dam.pivot(index='rte', columns='power_capacity', values='savings')
c = ax[0,0].pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[0,0], label='Savings (%)')
# draw a contour line at savings = 0
contours = ax[0,0].contour(pivot_table.columns, pivot_table.index, pivot_table.values, levels=[0], colors='red')
ax[0,0].clabel(contours, inline=True, fontsize=8, fmt='Savings = 0')
ax[0,0].set_xlabel('Power Capacity (kW)')
ax[0,0].set_ylabel('Round-Trip Efficiency (RTE)')
ax[0,0].set_title('DAM Cost Savings')

pivot_table = results_constant_power_dam.pivot(index='rte', columns='uptime', values='savings')
c = ax[0,1].pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[0,1], label='Savings (%)')
# draw a contour line at savings = 0
contours = ax[0,1].contour(pivot_table.columns, pivot_table.index, pivot_table.values, levels=[0], colors='red')
ax[0,1].clabel(contours, inline=True, fontsize=8, fmt='Savings = 0')
ax[0,1].set_xlabel('Uptime (%)')
ax[0,1].set_ylabel('Round-Trip Efficiency (RTE)')
ax[0,1].set_title('DAM Cost Savings')


pivot_table = results_constant_uptime_mef.pivot(index='rte', columns='power_capacity', values='savings')
c = ax[1,0].pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[1,0], label='Savings (%)')
# draw a contour line at savings = 0
contours = ax[1,0].contour(pivot_table.columns, pivot_table.index, pivot_table.values, levels=[0], colors='red')
ax[1,0].clabel(contours, inline=True, fontsize=8, fmt='Savings = 0')
ax[1,0].set_xlabel('Power Capacity (kW)')
ax[1,0].set_ylabel('Round-Trip Efficiency (RTE)')
ax[1,0].set_title('MEF Emissions Savings')


pivot_table = results_constant_power_mef.pivot(index='rte', columns='uptime', values='savings')
c = ax[1,1].pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[1,1], label='Savings (%)')
# draw a contour line at savings = 0
contours = ax[1,1].contour(pivot_table.columns, pivot_table.index, pivot_table.values, levels=[0], colors='red')
ax[1,1].clabel(contours, inline=True, fontsize=8, fmt='Savings = 0')
ax[1,1].set_xlabel('Uptime (%)')
ax[1,1].set_ylabel('Round-Trip Efficiency (RTE)')
ax[1,1].set_title('MEF Emissions Savings')
plt.tight_layout()

basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ext in ["png", "pdf", "svg"]:
    plt.savefig(os.path.join(basepath, "figures", ext, f"rte_analysis_{region}_month{month}.{ext}"), dpi=300)