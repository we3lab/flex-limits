# imports
import os
import numpy as np
import analysis.pricesignal as ps
from models.flexload_milp import flexloadMILP


region = "CAISO"
month = 1


def max_mef_savings(data, system_uptime, continuous_flex, baseload):

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


def max_tar_savings(
    data, 
    system_uptime, 
    continuous_flex, 
    baseload, 
    startdate_dt, 
    enddate_dt
):

    flex = flexloadMILP(
        baseload=baseload,
        cost_signal=data,
        costing_type="tariff",
        costing_path=None,
        emissions_type=None,
        emissions_path=None,
        flex_capacity=continuous_flex,
        rte=1.0,
        min_onsteps=max(int(len(data) * (system_uptime)), 1),
        uptime_equality=True,
        startdate_dt=startdate_dt,
        enddate_dt=enddate_dt,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_cost_savings()