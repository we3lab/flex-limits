# imports
import os, json
import numpy as np
import matplotlib.pyplot as plt

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

# import colors
with open(os.path.join(os.path.dirname(__file__), "colorscheme.json"), "r") as f:
    colors = json.load(f)
sys_colors=colors["examplesys_colors"]

fig, a = plt.subplots(figsize=(180 / 25.4/2, 120 / 25.4/2), layout="tight")

a.set_aspect("equal", adjustable="box")
a.set(
    xlim=(0, 100),
    xticks=np.arange(0, 101, 10),
    yticks=np.arange(0, 101, 10),
    ylim=(2, 100),
    aspect="equal",
)
a.set_xlabel("Power Capacity (%)", labelpad=1)
a.set_ylabel("System Uptime (%)", labelpad=1)

# power capacity, uptime
overlay = True
overlay_points = np.array([
    [0.99, 0.04],
    [0.01, 0.25],
    [0.5, 0.5],
    [0.25, 0.99]
])

if overlay:
    overlay_colors= [sys_colors["maxflex"], 
                    sys_colors["25uptime_0flex"],
                    sys_colors["50uptime_50flex"],
                    sys_colors["100uptime_25flex"]]

    # overlay points
    overlay_shapes = ["s", "^", "P", "o"]
    for i, point in enumerate(overlay_points):
        # calculate the color based on the index
        color = overlay_colors[i % len(overlay_colors)]
        # scatter the points
        a.scatter(
            point[0] * 100, 
            point[1] * 100, 
            marker=overlay_shapes[i],
            edgecolor="black",
            linewidth=1,
            color=color, 
            s=75, 
            clip_on=True
        )

for ext in ["png", "pdf", "svg"]:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_path = os.path.join(path, "figures", ext, f"designspace_plot.{ext}")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")