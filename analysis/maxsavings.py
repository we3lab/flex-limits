# imports
import os
import numpy as np
from eeco import costs
import analysis.pricesignal as ps
from models.flexload_milp import flexloadMILP
from models.flexload_milp import idxparam_value

def max_mef_savings(data, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for marginal emissions factor (MEF) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        emissions_signal=data,
        costing_type=None,
        costing_path=None,
        emissions_type="mef",
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=1.0,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()


def max_aef_savings(data, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for average emissions factor (AEF) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        emissions_signal=data,
        costing_type=None,
        costing_path=None,
        emissions_type="aef",
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=1.0,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()


def max_dam_savings(data, system_uptime, continuous_flex, baseload):
    """Calculate the maximum savings for day-ahead market (DAM) as a percentage."""
    flex = flexloadMILP(
        baseload=baseload,
        cost_signal=data,
        costing_type="dam",
        costing_path=None,
        emissions_type=None,
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=1.0,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_cost_savings()


def get_start_end(month):
    """Get the start and end datetimes based on the `month`"""
    if month == 12:
        start_dt = np.datetime64("2024-" + str(month) + "-01")
        end_dt = np.datetime64("2025-01-01")
    elif month > 9:
        start_dt = np.datetime64("2024-" + str(month) + "-01")
        end_dt = np.datetime64("2024-" + str(month+1) + "-01")
    elif month == 9:
        start_dt = np.datetime64("2024-09-01")
        end_dt = np.datetime64("2024-10-01")
    else:
        start_dt = np.datetime64("2024-0" + str(month) + "-01")
        end_dt = np.datetime64("2024-0" + str(month+1) + "-01")

    return (start_dt, end_dt)


def max_tariff_savings(
    data, 
    system_uptime, 
    continuous_flex, 
    baseload, 
    startdate_dt, 
    enddate_dt,
    uptime_equality=True,
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
        rte=1.0,
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
    consumption_data_dict = {"electric": idxparam_value(flex.model.net_facility_load) / 1000} # convert kW to MW

    total_flex_cost, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            desired_utility="electric",
            desired_charge_type=None,
            model=None,
            electric_consumption_units=u.MW
        )

    total_base_cost, _ = costs.calculate_cost(
            charge_dict,
            {"electric": flex.baseload}, # convert kW to MW
            resolution=resolution,
            desired_utility="electric",
            desired_charge_type=None,
            model=None,
            electric_consumption_units=u.MW
        )

    return (100 * (total_base_cost - total_flex_cost) / total_base_cost)