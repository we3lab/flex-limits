import numpy as np 
import pandas as pd
import os
from analysis.pricesignal import getmef, getaef, getdam
from models.flexload_milp import flexloadMILP, idxparam_value
from electric_emission_cost.units import u

def shadowcost_wholesale(
    region, 
    month, 
    system_uptime,
    continuous_flexibility,
    baseload=None,
    basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """
    Calculate the cost of savings from wholesale energy prices.

    Parameters:
    -----------
    region : str
        The region corresponding to ISO for which to calculate the cost of savings.
    month : str
        The month for which to calculate the cost of savings.
    system_uptime : float
        The percentage of time the system is available (0 to 1).
    continuous_flexibility : float
        The percentage deviation from the mean power (0 to 1).
    basepath : str, optional
        The base path to the data directory (default is the root of the repo).
    
    Returns:
    --------
    float
        The cost of emissions in usd/ton.
    """
    # Get the marginal emissions factor (MEF)
    mef = getmef(region, month, basepath)

    # Get the day ahead market price (LMP)
    dam = getdam(region, month, basepath)

    if baseload is None:
        # define a flat 1MW baseload 
        baseload = np.ones_like(mef)  
        
    # Calculate cost optimal while tracking emissions
    flex_cost = flexloadMILP(
                    baseload=baseload,
                    cost_signal=dam,
                    costing_type="dam",
                    costing_path=None,
                    emissions_type=None,
                    emissions_path=None,
                    flex_capacity=continuous_flexibility,
                    rte=1.0,
                    min_onsteps=max(int(len(mef) * (system_uptime)), 1),  # assuming daily cycles
                    uptime_equality=True,
                )
    flex_cost.build()
    flex_cost.solve()
    # Calculate the net facility load
    net_facility_load_costopt = idxparam_value(flex_cost.model.net_facility_load)

    # calculate the total cost and total emissions using the output power
    cost_optimal_cost = dam@net_facility_load_costopt
        # TODO: replace the unit conversion using pint
    cost_optimal_emissions = mef@net_facility_load_costopt* 0.001 # convert from $/kg to $/ton (metric)

    # Calculate emissions optimal while tracking cost
    flex_emissions = flexloadMILP(
        baseload=baseload,
        emissions_signal=mef,
        costing_type=None,
        costing_path=None,
        emissions_type="mef",
        emissions_path=None,
        flex_capacity=continuous_flexibility,
        cost_of_carbon=1.0,
        rte=1.0,
        min_onsteps=max(int(len(mef) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )
    flex_emissions.build()
    flex_emissions.solve()

    # Calculate the net facility load
    net_facility_load_emisopt = idxparam_value(flex_emissions.model.net_facility_load)
   
    # calculate the total cost and total emissions using the output power
    emissions_optimal_cost = dam@net_facility_load_emisopt
        # TODO: replace the unit conversion using pint
    emissions_optimal_emissions = mef@net_facility_load_emisopt* 0.001 # convert from $/kg to $/ton (metric)

    # Calculate the shadow price 
    shadow_price = -(cost_optimal_cost - emissions_optimal_cost) / (cost_optimal_emissions - emissions_optimal_emissions + 1e-8) 

    results = {
        "shadow_price_usd_ton": shadow_price,
        "cost_optimal_cost_usd": cost_optimal_cost,
        "emissions_optimal_cost_usd": emissions_optimal_cost,
        "cost_optimal_emissions_ton": cost_optimal_emissions,
        "emissions_optimal_emissions_ton": emissions_optimal_emissions
    }

    return results