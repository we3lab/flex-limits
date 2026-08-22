# flex-limits
This repository contains analysis to estimate the upper bound of benefits from energy flexibility with a particular focus on industrial loads. 


## Installation instructions
Navigate to your desired directory and run the following from the command line interface:
1. Clone the repository
```
git clone https://github.com/we3lab/flex-limits.git
```

2. Install packages
```
python setup.py install
```

## Cite this work

To cite this work, use the "Cite this repository" feature available on the right side of this repository page. Please reference the appropriate references from the list below:

For work related to calculating the upper bounds of benefits from flexible operation:
> Rao, A. K., Chapin, F.T., Musabandesu, E., Sakthivelu, A., Tucker, C.I., Wettermark, D., Mauter, M.S. How much can we save? Upper bound cost and emissions benefits from commercial and industrial load flexibility. _Under Review_. https://arxiv.org/abs/2511.14928


For work related to characterizing energy flexibility performance:
> Rao, A. K., Bolorinos, J., Musabandesu, E., Chapin, F. T., & Mauter, M. S. (2024). Valuing energy flexibility from water systems. Nature Water, 2(10), 1028-1037.

For work that uses the parameterization of tariffs:
> Chapin, F. T., Bolorinos, J., & Mauter, M. S. (2024). Electricity and natural gas tariffs at United States wastewater treatment plants. Scientific Data, 11(1), 113.

## Overview of this repository
```
flex_limits
|- analysis
|- data
|- models
|- paper_figures
```

**analysis**: Contains functions and classes that are used to run different analysis using the flexloadMILP model. 
- `acc_curve.py`: Computes the pareto curve between cost and emissions optimal objectives.
- `emissionscost.py`: Calculates the shadow cost of emissions abatement.
- `energy_capacity.py`: Evaluates the effect energy capacity as a function of uptime, power capacity, and RTE.
- `maxsavings.py`: Computes the maximum savings given energy flexibility characteristics and a region/month.
- `overlay_costs.py`: Calculates effective renewable energy credit prices + overlays that with the SCC on plots.
- `pricesignal.py`: Loads and reformats timeseries data from the **data** folder. Also generates synthetic time-varying baseload profiles via `synthetic_baseload`.
- `rte_analysis.py`: Calculates the effect of RTE<1 on the savings.
- `shadowcost.py`: Similar to emissionscost, an alternative method of calculating the shadow cost of emissions abatement.
- `tariff_features.py`: Extracts the rate structure of a tariff — how many demand and energy charges it has, whether its demand charges are coincident, how much price variation it offers — and itemizes the bill an inflexible load would pay under it. Pure cost evaluation via `eeco`, no optimization.

**data**: Contains cleaned data on electricity prices and emissions. 
- `aef/`: contains data on average emissions factors, sorted by region in month-hour-average format.
- `dam/`: folder not available. Day ahead market prices are excluded from the public repository but can be found via GridStatus. 
- `mef/`: contains data on average emissions factors, sorted by region in month-hour-average format.
- `offsets/`: contains information on estimates of renewable energy credit pricing and projections for the social cost of carbon. 
- `tariff/`: contains data on retail electricity tariffs (rate structures). An example is chosen for each region that is suitable for a 1MW load. A full list of tariffs is contained in the subfolder `bundled/`. A maintained list of industrial tariffs can be found in [this dataset](https://github.com/we3lab/industrial-electricity-tariffs).
- `tariff_wwtp/`: a subset of the `tariff/bundled/` folder that was used in an [initial study](https://www.nature.com/articles/s41597-023-02886-6) of electricity rates in the water sector. 

**models**: Contains a model file that represents the flexible load and is used for analysis.

- `flexload_milp.py`: contains the *flexloadMILP* class which builds a pyomo optimization model constrained based on the flexibility characteristics. 

**paper_figures**: Contains all code, figures, and data associated with figures in the published manuscript. 

- `code/`:
    - `acc_curve_wholesale_vs_tariff.py`: used to plot the pareto optimal curve between cost and emissions objectives. Manuscript figure 4.
    - `colorscheme.json`: Colors to be used across all plots.
    - `designspace_plot.py`: Used to map the uptime-power capacity space and plot example systems. Manuscript figure 1.
    - `energy_capacity_analysis.py`: Visualizes the effect energy capacity as a function of uptime, power capacity, and RTE.
    - `marginal_abatement_cost.py`: diagnostic that reads the same pareto curves as `acc_curve_wholesale_vs_tariff.py` and reports the cost of abatement at each point. It writes no figures — manuscript figures 4 c,d are the lower row of the 2x2 panel produced by `acc_curve_wholesale_vs_tariff.py`.
    - `maxsavings_boxplot.py`: plots the box plot associated with the range of savings from optimal flexibility. Manuscript figure 3. 
    - `maxsavings_contour.py`: plots the maximum savings as a function of uptime and power capacity. Manuscript figure 2. 
    - `rte_analysis.py`: plots the effects of RTE<1 on the maximum savings.
    - `shadowcosttariff_violinplot.py`: plots the cost of abatement associated with flexibility when considering tariff electricity pricing. Manuscript figure 5b. 
    - `tariff_cluster_analysis.py`: clusters the bundled tariffs by rate structure and compares how much value a flexible load extracts from each archetype. Supporting information. Runs no optimization — it reuses the `cost_optimal_cost_usd` already solved in `processed_data/shadowcost_tariff_mef/` and pairs it with an inflexible reference bill obtained by evaluating each tariff against a flat 1 MW load.
    - `timevarying_baseload_example.py`: plots an example of optimally flexible operation of a time-varying baseload against the day-ahead market price signal, alongside the power capacity band.
    - `shadowcostwholesale_boxplot.py`: plots the cost of abatement associated with flexibility when considering day-ahead market electricity pricing. Manuscript figure 5a. 
- `figures/`: contains subfolders for 3 supported image file types: `pdf`, `svg`, and `png`. The top level of each holds **only the manuscript figures**; supporting-information figures go to `figures/SI/{pdf,svg,png}/`, and retired or superseded renders go to `figures/ARCHIVE/`. Scripts pick their destination through `code/figpaths.py::figure_dir`, so a figure moves between the two by flipping one `si=` argument rather than by being copied around. Where a supporting figure needs more explanation than its axes can carry, the caption text sits beside the renders as `figures/SI/<figure_name>_caption.md` — see `figures/SI/tariff_cluster_analysis_caption.md`.
-  `processed_data/`: Contains data associated with each of the `figures/`. `processed_data/ARCHIVE/README.md` records which generated datasets are current, which are retired and why, and the cheap regression tests used to tell them apart — read it before rebuilding a figure from anything in here.


## Funding Acknowledgements
This work is being conducted as part of the [National Alliance for Water Innovation
(NAWI)](https://www.nawihub.org/) with support through the U.S. Department of Energy’s [Advanced
Manufacturing Office](https://www.energy.gov/eere/amo/advanced-manufacturing-office).
