# imports
import os, json
import numpy as np
import matplotlib.pyplot as plt
from analysis.pricesignal import getmef, getdam

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
month = 6

mef = getmef(region, month)
dam = getdam(region, month)
hours = np.arange(len(mef))

fig, ax1 = plt.subplots(figsize=(90 / 25.4, 60 / 25.4), layout="tight")

color_mef = incentive_colors["mef"]
color_dam = incentive_colors["dam"]

ax1.step(hours, dam, color=color_dam, label="Cost", where="post")
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("DAM price (USD/MWh)", color=color_dam)
ax1.tick_params(axis="y", labelcolor=color_dam)
ax1.set_xticks(np.arange(0, 24, 3))
ax1.set_xlim(0, 23)
ax1.set_yticks(np.arange(0, 61, 10))

ax2 = ax1.twinx()
ax2.step(hours, mef, color=color_mef, label="Emissions", where="post")
ax2.set_ylabel("MEF (kg CO$_2$eq/MWh)", color=color_mef)
ax2.tick_params(axis="y", labelcolor=color_mef)
# ax2.set_yticks(np.arange(0, 601, 100))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

figpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fig_fmt in ["png", "pdf", "svg"]:
    plt.savefig(
        os.path.join(figpath, f"figures/{fig_fmt}/hourly_pricesignal_{region}_month{month}.{fig_fmt}"),
        dpi=400,
        bbox_inches="tight",
    )

plt.show()
