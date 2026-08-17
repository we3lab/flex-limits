# imports
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.pricesignal import getdam, synthetic_baseload
from analysis.maxsavings import max_dam_savings
from models.flexload_milp import idxparam_value

# load color scheme
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)

incentive_colors = colors["incentive_colors"]

# plotting defaults
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

region = "CAISO"
month = 7
uptime = 0.75
power_capacity = 0.25

# time-varying baseload: low overnight, peaking at 2 pm
dam = getdam(region, month)
baseload = synthetic_baseload(n=len(dam))

# optimally flexible operation against the day-ahead market price signal
pct_savings, flex = max_dam_savings(
    dam, uptime, power_capacity, baseload, return_model=True
)
flexload = idxparam_value(flex.model.net_facility_load)

# power capacity band enforced on the load while it is on, read off the solved model
band_lo = idxparam_value(flex.model.min_contload)
band_hi = idxparam_value(flex.model.max_contload)

# duplicate the final value so that post-step lines and fills span the full day
hours = np.arange(len(dam) + 1)
edge = lambda y: np.append(y, y[-1])

fig, ax = plt.subplots(
    2, 1, figsize=(90 / 25.4, 80 / 25.4), sharex=True, layout="tight"
)

# (a) price signal
ax[0].step(hours, edge(dam), color=incentive_colors["dam"], where="post")
ax[0].set_ylabel("DAM price\n(USD/MWh)", labelpad=1)
ax[0].set_ylim(0, None)

# (b) load trajectories
band_color = colors["other"]["scc"]
ax[1].fill_between(
    hours,
    edge(band_lo),
    edge(band_hi),
    step="post",
    color=band_color,
    alpha=0.15,
    linewidth=0,
    label="Power capacity band",
)
for bound in (band_lo, band_hi):
    ax[1].step(hours, edge(bound), color=band_color, linestyle=":", where="post")

ax[1].step(hours, edge(baseload), color=band_color, where="post", label="Baseline")
ax[1].step(
    hours,
    edge(flexload),
    color=incentive_colors["dam"],
    where="post",
    label="Optimally flexible",
)
ax[1].set_ylabel("Power (MW)", labelpad=1)
# a small margin below zero keeps the off-state segments off the axis spine
ax[1].set_ylim(-0.07 * band_hi.max(), 1.32 * band_hi.max())
ax[1].set_xlabel("Hour of day", labelpad=1)
ax[1].legend(loc="upper left", frameon=False, handlelength=1.5)
ax[1].text(
    0.98,
    0.97,
    "{:.0f}% uptime, {:.0f}% power capacity\n{:.1f}% cost savings".format(
        100 * uptime, 100 * power_capacity, pct_savings
    ),
    transform=ax[1].transAxes,
    va="top",
    ha="right",
)

ax[1].set_xlim(0, len(dam))
ax[1].set_xticks(np.arange(0, len(dam) + 1, 3))

for i, a in enumerate(ax):
    a.text(
        -0.15,
        1.04,
        "{}.".format("ab"[i]),
        transform=a.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="left",
    )
fig.align_ylabels(ax)

figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fig_fmt in ["png", "pdf", "svg"]:
    plt.savefig(
        os.path.join(
            figpath,
            f"figures/{fig_fmt}/timevarying_baseload_example_{region}_month{month}.{fig_fmt}",
        ),
        dpi=400,
        bbox_inches="tight",
    )

datapath = os.path.join(figpath, "processed_data", "timevarying_baseload_example")
os.makedirs(datapath, exist_ok=True)
pd.DataFrame(
    {
        "hour": np.arange(len(dam)),
        "dam": dam,
        "baseload": baseload,
        "flexload": flexload,
        "band_lo": band_lo,
        "band_hi": band_hi,
    }
).to_csv(os.path.join(datapath, f"{region}_month{month}.csv"), index=False)

plt.show()
