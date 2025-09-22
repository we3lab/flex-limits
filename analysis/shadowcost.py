from models.flexload_milp import flexloadMILP
from pyomo.environ import value, Param, Constraint, Expression
import numpy as np
import pandas as pd 
import os 
import calendar
from analysis.pricesignal import getmef, getaef, getdam, gettariff
from analysis.acc_curve import acc_curve
from analysis import maxsavings as ms


def shadow_cost(system_uptime, 
                power_capacity, 
                region, 
                month,
                uptime_equality = True,
                abatement_frac=1.0,
                baseload = None,
                emissions_type="mef",
                costing_type="dam",
                year=2025, 
                tol=1e-8):
    """
    Calculate the cost of carbon abatement for a flexible load system.
    Parameters:
    -----------
    system_uptime : float
        Fraction of time the system is operational (0 < system_uptime <= 1).
    power_capacity : float
        Fraction of power that can be flexed at a given moment (0 <= power_capacity <= 1).
    region : str
        The region corresponding to ISO for which to calculate the cost of carbon abatement.
    month : int
        The month (1-12) for which to calculate the cost of carbon abatement.
    uptime_equality : bool
        If True, enforces the uptime as an equality constraint over the horizon.
        If False, enforces it as a minimum constraint.
    abatement_frac : float
        Fraction of maximum emissions abatement to achieve (0 < abatement_frac <= 1).
    baseload : array-like, optional
        Baseline load profile. If None, defaults to a constant load of 1MW.
    emissions_type : str
        Type of emission factor: "mef" or "aef".
    costing_type : str
        Type of cost signal: "dam" or "tariff".
    """

    # get data info
    if emissions_type == "mef":
        emissions_signal = getmef(region, month)
    elif emissions_type == "aef":
        emissions_signal = getaef(region, month)
    else:
        raise ValueError("emissions_type must be 'mef' or 'aef'")
    
    if costing_type == "dam":
        cost_signal = getdam(region, month)
    elif costing_type == "tariff":
        cost_signal = gettariff(region, full_list=False)
        _, num_days = calendar.monthrange(year, month = month) 
        emissions_signal = np.tile(emissions_signal, num_days)

    startdate_dt, enddate_dt = ms.get_start_end(month)
    
    curve = acc_curve(
        baseload=np.ones_like(emissions_signal) if baseload is None else baseload,
        min_onsteps=max(int(len(emissions_signal) * (system_uptime)), 1),
        flex_capacity=power_capacity,
        emissions_signal=emissions_signal,
        emissions_type=emissions_type,
        cost_signal=cost_signal,
        costing_type=costing_type,
        startdate_dt=startdate_dt,
        enddate_dt=enddate_dt,
        uptime_equality=uptime_equality,
    )
    
    curve.calc_emissions_optimal()
    curve.calc_cost_optimal()

    if abatement_frac == 1.0:
        shadow_cost = (curve.cost_optimal_cost - curve.emissions_optimal_cost) / (curve.emissions_optimal_emissions / 1000 - curve.cost_optimal_emissions / 1000 + tol)
    elif abatement_frac < 1.0:
        # add parameters and constraints to model
        allowable_emissions = (curve.cost_optimal_emissions - curve.emissions_optimal_emissions) * (1 - abatement_frac) + curve.emissions_optimal_emissions
        curve.model.allowable_emissions = Param(initialize=allowable_emissions, mutable=True)
        @curve.model.Constraint()
        def emissions_limit_rule(m):
            return m.total_flex_emissions_signal <= m.allowable_emissions
        
        # re-solve cost optimal
        model, results = curve.solve(threads = threads, max_iter=100, descent_stepsize=0.1)

                # extract results from cost optimal solution
        if curve.costing_type == "dam":
            cost = sum(idxparam_value(curve.model.net_facility_load) * curve.model.cost_signal) 

        elif curve.costing_type == "tariff":
            # (1) get the charge dictionary
            curve.charge_dict = costs.get_charge_dict(
                curve.startdate_dt, curve.enddate_dt, curve.cost_signal, resolution=resolution
            )
            # (2) set up consumption data dictionary
            consumption_data_dict = {"electric": idxparam_value(curve.model.net_facility_load)}

            # (3) calculate the costs
            cost, _ = costs.calculate_cost(
                curve.charge_dict,
                consumption_data_dict,
                resolution=resolution,
                desired_utility="electric",
                desired_charge_type=None,
                model=None,
                electric_consumption_units=u.MW
            )
        emissions = sum(idxparam_value(curve.model.net_facility_load) * curve.model.emissions_signal)

        # check that we met the emissions constraint
        assert hasattr(model, 'emissions_limit_rule'), "Emissions limit not added to model."
        if curve.cost_optimal_emissions - allowable_emissions > tol:
            raise ValueError("Could not meet emissions constraint. Try increasing system_uptime or power_capacity.")
        shadow_cost = (curve.cost_optimal_cost - cost) / (emission / 1000 - curve.cost_optimal_emissions / 1000 + tol)

    if abatement_frac <= 0 or abatement_frac > 1:
        raise ValueError("abatement_frac must be between 0 and 1")

    return shadow_cost
    
