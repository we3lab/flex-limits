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
> Rao, A. K., Chapin, F.T., Musabandesu, E., Sakthivelu, A., Tucker, C.I., Wettermark, D., Mauter, M.S. How much can we save? Upper bound cost and emissions benefits from commercial and industrial load flexibility. Manuscript in Progress.


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
- `pricesignal.py`: Loads and reformats timeseries data from the **data** folder. 
- `rte_analysis.py`: Calculates the effect of RTE<1 on the savings.
- `shadowcost.py`: Similar to emissionscost, an alternative method of calculating the shadow cost of emissions abatement.

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
    - `marginal_abatement_cost.py`: used to plot the cost of abatement associated with points on the pareto curve. Manuscript figures 4 c,d.
    - `maxsavings_boxplot.py`: plots the box plot associated with the range of savings from optimal flexibility. Manuscript figure 3. 
    - `maxsavings_contour.py`: plots the maximum savings as a function of uptime and power capacity. Manuscript figure 2. 
    - `rte_analysis.py`: plots the effects of RTE<1 on the maximum savings.
    - `shadowcost_contour`: plots the cost of abatement as a function of uptime, power capacity, and fractional abatement.
    - `shadowcosttariff_violinplot.py`: plots the cost of abatement associated with flexibility when considering tariff electricity pricing. Manuscript figure 5b. 
    - `shadowcostwholesale_boxplot.py`: plots the cost of abatement associated with flexibility when considering day-ahead market electricity pricing. Manuscript figure 5a. 
- `figures/`: contains subfolders for 3 supported image file types: `pdf`, `svg`, and `png`. Each contains several versions of the plots generated by the aforementioned code.
-  `processed_data/`: Contains data associated with each of the `figures/`.


## Funding Acknowledgements
This work is being conducted as part of the [National Alliance for Water Innovation
(NAWI)](https://www.nawihub.org/) with support through the U.S. Department of Energy’s [Advanced
Manufacturing Office](https://www.energy.gov/eere/amo/advanced-manufacturing-office).