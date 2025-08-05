# imports
import os
import numpy as np
from electric_emission_cost import costs
import analysis.pricesignal as ps
from analysis.maxsavings import get_start_end
from models.flexload_milp import flexloadMILP
from models.flexload_milp import idxparam_value

def mef_savings(data, rte, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for marginal emissions factor (MEF) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        emissions_signal=data,
        costing_type=None,
        costing_path=None,
        emissions_type="mef",
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=rte,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()

def aef_savings(data, rte, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for average emissions factor (AEF) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        emissions_signal=data,
        costing_type=None,
        costing_path=None,
        emissions_type="aef",
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=rte,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()

def dam_savings(data, rte, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for day-ahead market (DAM) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        cost_signal=data,
        costing_type="dam",
        costing_path=None,
        emissions_type=None,
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=rte,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_cost_savings()

def tariff_savings(
    data, 
    rte,
    system_uptime, 
    continuous_flex, 
    baseload, 
    startdate_dt, 
    enddate_dt,
    uptime_equality=False,
    resolution="1h"
):
    """Calculate the maximum savings for a tariff as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        cost_signal=data,
        costing_type="tariff",
        costing_path=None,
        emissions_type=None,
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=rte,
        min_onsteps=max(int(len(baseload) * (system_uptime)), 1),
        uptime_equality=uptime_equality,
        startdate_dt=startdate_dt,
        enddate_dt=enddate_dt,
    )

    flex.build()
    flex.solve()

    # (1) get the charge dictionary
    charge_dict = costs.get_charge_dict(
        startdate_dt, enddate_dt, data, resolution=resolution
    )

    # (2) set up consumption data dictionary
    consumption_data_dict = {"electric": idxparam_value(flex.model.flexload)}

    energy_flex_cost, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            desired_utility="electric",
            desired_charge_type="energy",
            model=None,
        )
    demand_flex_cost, _ = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        resolution=resolution,
        desired_utility="electric",
        desired_charge_type="demand",
        model=None,
    )

    total_base_cost = flex.model.total_base_cost_signal
    total_flex_cost = demand_flex_cost + energy_flex_cost
    return (100 * (total_base_cost - total_flex_cost) / total_base_cost)