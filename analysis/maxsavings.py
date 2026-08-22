# imports
import os
from functools import lru_cache
import numpy as np
from eeco import costs
from eeco.units import u
import analysis.pricesignal as ps
from models.flexload_milp import flexloadMILP
from models.flexload_milp import idxparam_value

def max_mef_savings(data, system_uptime, continuous_flex, baseload, threads=10):
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
    flex.solve(threads=threads)

    return flex.model.pct_emissions_savings()


def max_aef_savings(data, system_uptime, continuous_flex, baseload, threads=10):
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
    flex.solve(threads=threads)

    return flex.model.pct_emissions_savings()


def max_dam_savings(data, system_uptime, continuous_flex, baseload, return_model=False, threads=10):
    """Calculate the maximum savings for day-ahead market (DAM) as a percentage.

    If `return_model` is True, the solved `flexloadMILP` is returned alongside the savings
    so that callers can inspect the optimal trajectory.
    """
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
    flex.solve(threads=threads)

    if return_model:
        return flex.model.pct_cost_savings(), flex

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
    resolution="1h",
    threads=10
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
    flex.solve(threads=threads)

    # (1) get the charge dictionary
    charge_dict = costs.get_charge_dict(
        startdate_dt, enddate_dt, data, resolution=resolution
    )

    # (2) set up consumption data dictionary
    # both trajectories are in MW, which is declared to eeco below; scaling only one of
    # them here would make the flexible and baseline costs incomparable
    consumption_data_dict = {"electric": idxparam_value(flex.model.net_facility_load)}

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
            {"electric": flex.baseload},
            resolution=resolution,
            desired_utility="electric",
            desired_charge_type=None,
            model=None,
            electric_consumption_units=u.MW
        )

    return (100 * (total_base_cost - total_flex_cost) / total_base_cost)


@lru_cache(maxsize=None)
def get_signals(region, month):
    """Load the MEF, AEF, DAM and tariff signals for a region-month.

    Cached because `max_savings_point` is the unit of work fanned out across a
    uptime/power-capacity grid, and every point in a region-month needs the same inputs.
    """
    startdate_dt, enddate_dt = get_start_end(month)
    return (
        ps.getmef(region, month),
        ps.getaef(region, month),
        ps.getdam(region, month),
        ps.gettariff(region, full_list=False),
        startdate_dt,
        enddate_dt,
    )


def max_savings_point(region, month, system_uptime, continuous_flex, threads=1):
    """Maximum MEF, AEF, DAM and tariff savings at one (uptime, power capacity) point.

    Lives here rather than in the plotting script so that parallel workers can import it
    and load their own inputs, instead of having the signals shipped with every task.
    """
    mef, aef, dam, tariff, startdate_dt, enddate_dt = get_signals(region, month)
    baseload = np.ones_like(mef)  # 1MW flat baseline load

    # tariffs require a full month of data to properly incorporate monthly demand charges
    month_length = int((enddate_dt - startdate_dt) / np.timedelta64(1, "h"))

    return (
        max_mef_savings(mef, system_uptime, continuous_flex, baseload, threads=threads),
        max_aef_savings(aef, system_uptime, continuous_flex, baseload, threads=threads),
        max_dam_savings(dam, system_uptime, continuous_flex, baseload, threads=threads),
        max_tariff_savings(
            tariff,
            system_uptime,
            continuous_flex,
            np.ones(month_length),
            uptime_equality=True,
            startdate_dt=startdate_dt,
            enddate_dt=enddate_dt,
            threads=threads,
        ),
    )