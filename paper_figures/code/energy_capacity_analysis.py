# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from analysis.energy_capacity import calc_energy_capacity

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
region="SPP"
month=1
def solve_energy_capacity_dam(uptime, power_capacity, rte):
    ecap = calc_energy_capacity(uptime=uptime,
                       power_capacity=power_capacity,
                       region=region,
                       month=month,
                       rte=rte,
                       signal="cost")
    return pd.DataFrame({"power_capacity": [power_capacity],
                            "uptime": [uptime],
                             "energy_capacity": [ecap]})

def solve_energy_capacity_mef(uptime, power_capacity, rte):
    ecap= calc_energy_capacity(uptime=uptime,
                       power_capacity=power_capacity,
                       region=region,
                       month=month,
                       rte=rte,
                       signal="cost")
    return pd.DataFrame({"uptime": [uptime],
                            "power_capacity": [power_capacity],
                             "energy_capacity": [ecap]})
# %%
tasks = []
for power_capacity in np.arange(0.1, 1.1, 0.1):
    for uptime in np.arange(0.5, 1.05, 0.05):
        tasks.append((uptime, power_capacity))
results_rte1 = Parallel(n_jobs=20)(delayed(solve_energy_capacity_dam)(uptime, power_capacity, 1.0) for uptime, power_capacity in tasks)
results_rte1 = pd.concat(results_rte1, ignore_index=True)

results_rte07 = Parallel(n_jobs=20)(delayed(solve_energy_capacity_mef)(uptime, power_capacity, 0.5) for uptime, power_capacity in tasks)
results_rte07 = pd.concat(results_rte07, ignore_index=True)

# %%
fig, ax = plt.subplots(1,2, figsize=(180 / 25.4, 100 / 25.4))
pivot_table = results_rte1.pivot(index='uptime', columns='power_capacity', values='energy_capacity')
c = ax[0].pcolormesh(pivot_table.columns*100, pivot_table.index*100, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[0], label='Energy Capacity (%)')
ax[0].set_xlabel('Power Capacity (%)')
ax[0].set_ylabel('Uptime (%)')
ax[0].set_title('RTE=100%')

pivot_table = results_rte07.pivot(index='uptime', columns='power_capacity', values='energy_capacity')
c = ax[1].pcolormesh(pivot_table.columns*100, pivot_table.index*100, pivot_table.values, shading='auto', cmap='viridis')
fig.colorbar(c, ax=ax[1], label='Energy Capacity (%)')
ax[1].set_xlabel('Power Capacity (%)')
ax[1].set_ylabel('Uptime (%)')
ax[1].set_title('RTE=70%')
plt.tight_layout()

basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ext in ["png", "pdf", "svg"]:
    plt.savefig(os.path.join(basepath, "figures", ext, f"energy_capacity.{ext}"), dpi=300)