import numpy as np 
import pandas as pd
import os
import calendar
from pyomo.environ import Param, Var, NonNegativeReals, Constraint
from analysis.pricesignal import getmef, getaef, getdam, gettariff
from analysis import maxsavings as ms
from analysis.acc_curve import acc_curve
from models.flexload_milp import flexloadMILP, idxparam_value
from eeco import costs
from eeco.units import u

def get_hourly_average_emission_factors(region, emission_type="mef"):
    """
    Calculate hourly average emission factors for a region across all months.
    
    Parameters:
    -----------
    region : str
        The region/ISO for which to get emission factors.
    emission_type : str
        Type of emission factor: "mef" or "aef".
    
    Returns:
    --------
    tuple
        (min_emission_factor, max_emission_factor) in ton/MWh
    """
    monthly_hourly_avg_emission_kg_per_mwh = []
    
    for month in range(1, 13):
        if emission_type.lower() == "mef":
            emission_data = getmef(region, month)
        elif emission_type.lower() == "aef":
            emission_data = getaef(region, month)
        else:
            raise ValueError("emission_type must be 'mef' or 'aef'")
        
        # Calculate hourly average for this month
        hourly_avg_emission = np.mean(emission_data)
        monthly_hourly_avg_emission_kg_per_mwh.append(hourly_avg_emission)
    
    # Convert to ton/MWh
    monthly_hourly_avg_emission_ton_per_mwh = np.array(monthly_hourly_avg_emission_kg_per_mwh) / 1000
    
    return monthly_hourly_avg_emission_ton_per_mwh

def shadowcost_wholesale(
    region, 
    month, 
    system_uptime,
    continuous_flexibility,
    abatement_fraction=1.0,
    baseload=None,
    basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    emissions_type="mef",
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
    if emissions_type.lower() == "mef":
        emis = getmef(region, month, basepath)
    elif emissions_type.lower() == "aef":
        emis = getaef(region, month, basepath)
    else:
        raise ValueError("emissions_type must be 'mef' or 'aef'")

    # Get the day ahead market price (LMP)
    dam = getdam(region, month, basepath)

    if baseload is None:
        # define a flat 1MW baseload 
        baseload = np.ones_like(emis)  

    startdate_dt, enddate_dt = ms.get_start_end(month)
        
    # Calculate cost optimal while tracking emissions
    acc = acc_curve(
                    baseload=baseload,
                    cost_signal=dam,
                    costing_type="dam",
                    emissions_signal=emis,
                    emissions_type=emissions_type.lower(),
                    flex_capacity=continuous_flexibility,
                    min_onsteps=max(int(len(emis) * (system_uptime)), 1),  # assuming daily cycles
                    uptime_equality=True,
                    startdate_dt=startdate_dt,
                    enddate_dt=enddate_dt,
                )
    acc.calc_cost_optimal()
    acc.calc_emissions_optimal()

    c_star = acc.cost_optimal_cost
    e_star = acc.cost_optimal_emissions / 1000


    acc.model.weight_of_cost = 1.0
    acc.model.cost_of_carbon = 0.0
    acc.model.abatement_fraction = Param(initialize=abatement_fraction, mutable=True)
    acc.model.allowable_emissions = Var(within = NonNegativeReals)
    @acc.model.Constraint()
    def allowable_emissions_rule(m):
        return m.allowable_emissions == (acc.cost_optimal_emissions - acc.emissions_optimal_emissions) * (1 - abatement_fraction) + acc.emissions_optimal_emissions
    
    @acc.model.Constraint()
    def emissions_limit_rule(m):
        return m.total_flex_emissions_signal <= m.allowable_emissions

    # recalculate the cost and emissions at the new point on the curve
    model, results = acc.solve(acc.model)

    assert results.solver.termination_condition == 'optimal', "Solver did not find an optimal solution."
    assert model.find_component('allowable_emissions_rule') is not None, "Model does not have 'net_facility_load' component."

    print(f"Abatement fraction: {acc.model.abatement_fraction.value}, New allowable emissions: {acc.model.allowable_emissions.value}")

    c_k = sum(idxparam_value(acc.model.net_facility_load) * dam)
    e_k = sum(idxparam_value(acc.model.net_facility_load) * emis) / 1000  # convert from kg to ton (metric)


    # Calculate the shadow price 
    shadow_price = -(c_star - c_k) / (e_star - e_k + 1e-8) 

    results = {
        "shadow_price_usd_ton": shadow_price,
        "cost_optimal_cost_usd": c_star,
        "emissions_optimal_cost_usd": c_k,
        "cost_optimal_emissions_ton": e_star,
        "emissions_optimal_emissions_ton": e_k
    }

    return results



def shadowcost_tariff(
    region, 
    tariff_data, 
    # electricity_costing_CWNS_No, #change this for larger dataset + add to parameters below 
    # tariff_costing_path, 
    month, 
    system_uptime,
    continuous_flexibility,
    baseload=None,
    basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    uptime_equality = True, 
    threads = 10, 
    year = 2023,
    emissions_type="aef",
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
    # get number of days, start date, and end date 
    _, num_days = calendar.monthrange(year, month = month) 
    startdate_dt, enddate_dt = ms.get_start_end(month)

    # Get the emissions factor (MEF)
    if emissions_type.lower() == "mef":
        emis = getmef(region, month, basepath)
    elif emissions_type.lower() == "aef":
        emis = getaef(region, month, basepath)
    else:
        raise ValueError("emissions_type must be 'mef' or 'aef'")


    emis = np.tile(emis, num_days)

    if baseload is None:
        # define a flat 1MW baseload 
        baseload = np.ones_like(emis) 
        
    # Calculate cost optimal while tracking emissions
    flex_cost = flexloadMILP(
                    baseload=baseload,
                    cost_signal=tariff_data,
                    costing_type="tariff",
                    costing_path = None, 
                    startdate_dt=startdate_dt,
                    enddate_dt=enddate_dt,
                    emissions_type=None,
                    emissions_path=None,
                    flex_capacity=continuous_flexibility,
                    rte=1.0,
                    min_onsteps=max(int(len(emis) * (system_uptime)), 1),  # assuming daily cycles
                    uptime_equality=uptime_equality,
)
    flex_cost.build()
    flex_cost.solve(threads = threads)
    # Calculate the net facility load
    net_facility_load_costopt = idxparam_value(flex_cost.model.net_facility_load)

    # (1) get the charge dictionary
    charge_dict = costs.get_charge_dict(
        startdate_dt, enddate_dt, tariff_data, resolution="1h"
    )

    # (2) set up consumption data dictionary
    consumption_data_dict_costopt = {"electric": net_facility_load_costopt}

    cost_optimal_cost, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict_costopt,
            resolution="1h",
            desired_utility="electric",
            desired_charge_type=None,
            model=None,
            electric_consumption_units=u.MW
        )

        # TODO: replace the unit conversion using pint
    cost_optimal_emissions = emis@net_facility_load_costopt* 0.001 # convert from $/kg to $/ton (metric)

    # Calculate emissions optimal while tracking cost

    flex_emissions = flexloadMILP(
        baseload=baseload,
        emissions_signal=emis,
        costing_type=None,
        costing_path=None,
        emissions_type=emissions_type,
        emissions_path=None,
        flex_capacity=continuous_flexibility,
        cost_of_carbon=1.0,
        rte=1.0,
        min_onsteps=max(int(len(emis) * (system_uptime)), 1),  # assuming daily cycles
        uptime_equality=True,
    )

    flex_emissions.build()
    flex_emissions.solve(threads = threads)

    # Calculate the net facility load
    net_facility_load_emisopt = idxparam_value(flex_emissions.model.net_facility_load) / 1000 # convert kW to MW
   
    # (2) set up consumption data dictionary
    consumption_data_dict_emisopt = {"electric": net_facility_load_emisopt}

    emissions_optimal_cost, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict_emisopt,
            resolution="1h",
            desired_utility="electric",
            desired_charge_type=None,
            model=None,
            electric_consumption_units=u.MW
        )

        # TODO: replace the unit conversion using pint
    emissions_optimal_emissions = emis@net_facility_load_emisopt* 0.001 # convert from kg to ton (metric)

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