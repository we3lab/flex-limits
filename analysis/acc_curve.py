from models.flexload_milp import flexloadMILP, idxparam_value
from pyomo.environ import value, Param, Constraint, Var, NonNegativeReals
import numpy as np
import pandas as pd 
import os 
import calendar
from analysis.pricesignal import getmef, getaef, getdam, gettariff
import analysis.maxsavings as ms
from electric_emission_cost import costs
from electric_emission_cost.units import u

class acc_curve(flexloadMILP):
    def __init__(self, baseload, min_onsteps, flex_capacity, emissions_signal, emissions_type, 
                 cost_signal, costing_type, startdate_dt, enddate_dt, uptime_equality):
        # call the parent class constructor
        super().__init__(        
            baseload=baseload,
            flex_capacity=flex_capacity,
            min_onsteps=min_onsteps,  # assuming daily cycles
            emissions_signal=emissions_signal,
            emissions_type=emissions_type,
            emissions_path = None, 
            cost_signal=cost_signal,
            costing_type=costing_type,
            costing_path = None, 
            startdate_dt=startdate_dt,
            enddate_dt=enddate_dt,
            cost_of_carbon=1.0,
            weight_of_cost=1.0, 
            rte=1.0,
            uptime_equality=uptime_equality,
            )

        # define some attributes to store the results
        self.cost_optimal_emissions = 0
        self.cost_optimal_cost = 0
        self.emissions_optimal_emissions = 0 
        self.emissions_optimal_cost = 0
        self.baseline_emissions = 0
        self.baseline_cost = 0

        # define the base model object as an attribute of the acc_curve class
        self.model = self.build()
        return

    def calc_baseline(self, threads=10, resolution="1h"):
        self.baseline_emissions = sum(idxparam_value(self.model.net_facility_load) * self.model.emissions_signal)
        
        if self.costing_type == "dam":
            self.baseline_cost = sum(idxparam_value(self.model.net_facility_load) * self.model.cost_signal)
        
        elif self.costing_type == "tariff":
            # (1) get the charge dictionary
            charge_dict = costs.get_charge_dict(
                self.startdate_dt, self.enddate_dt, self.cost_signal, resolution=resolution
            )
            # (2) set up consumption data dictionary
            consumption_data_dict = {"electric": self.baseload}

            # (3) calculate the costs
            self.baseline_cost, _ = costs.calculate_cost(
                charge_dict,
                consumption_data_dict,
                resolution=resolution,
                desired_utility="electric",
                desired_charge_type=None,
                model=None,
                electric_consumption_units=u.MW
            )

    def calc_emissions_optimal(self, threads = 10, resolution="1h"):
        self.model.weight_of_cost = 0 
        self.model.cost_of_carbon = 1
        (self.model, results) = self.solve(threads = threads, max_iter=100, descent_stepsize=0.1)
        
        # extract results from cost optimal solution
        if self.costing_type == "dam":
            self.emissions_optimal_cost = sum(idxparam_value(self.model.net_facility_load) * self.model.cost_signal)
        
        elif self.costing_type == "tariff":
            # (1) get the charge dictionary
            self.charge_dict = costs.get_charge_dict(
                self.startdate_dt, self.enddate_dt, self.cost_signal, resolution=resolution
            )
            # (2) set up consumption data dictionary
            consumption_data_dict = {"electric": idxparam_value(self.model.net_facility_load)} 

            # (3) calculate the costs
            self.emissions_optimal_cost, _ = costs.calculate_cost(
                self.charge_dict,
                consumption_data_dict,
                resolution=resolution,
                desired_utility="electric",
                desired_charge_type=None,
                model=None,
                electric_consumption_units=u.MW
            )

        self.emissions_optimal_emissions = sum(idxparam_value(self.model.net_facility_load) * self.model.emissions_signal) 

        return self.emissions_optimal_cost, self.emissions_optimal_emissions
    
    def calc_cost_optimal(self, threads = 10, resolution="1h"):
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0
        (self.model, results) = self.solve(threads = threads, max_iter=100, descent_stepsize=0.1)

        # extract results from cost optimal solution
        if self.costing_type == "dam":
            self.cost_optimal_cost = sum(idxparam_value(self.model.net_facility_load) * self.model.cost_signal) 

        elif self.costing_type == "tariff":
            # (1) get the charge dictionary
            self.charge_dict = costs.get_charge_dict(
                self.startdate_dt, self.enddate_dt, self.cost_signal, resolution=resolution
            )
            # (2) set up consumption data dictionary
            consumption_data_dict = {"electric": idxparam_value(self.model.net_facility_load)}

            # (3) calculate the costs
            self.cost_optimal_cost, _ = costs.calculate_cost(
                self.charge_dict,
                consumption_data_dict,
                resolution=resolution,
                desired_utility="electric",
                desired_charge_type=None,
                model=None,
                electric_consumption_units=u.MW
            )

        else:
            raise ValueError(f"Unknown costing type: {self.costing_type}")


        self.cost_optimal_emissions = sum(idxparam_value(self.model.net_facility_load) * self.model.emissions_signal)
        return self.cost_optimal_cost, self.cost_optimal_emissions

    def build_pareto_front(self, stepsize=5, rel_tol = 1e-5, threads = 10, savepath = None):
        """
        **Description**:

        Function sweeps the pareto curve to find the relationship between the carbon cost and the fraction of potential emissions savings.

        **Parameters**:

            stepsize (float - optional): the resolution of the pareto curve sweep in units of carbon cost
            rel_tol (float - optional): the relative tolerance to which the emissions optimal solution is considered reached       
            savepath (str - optional): the path to save the results to a csv file

        **Returns**:

            pareto_front (DataFrame): a dataframe containing the carbon cost, total electrical emissions, electricity cost, alignment cost, and alignment fraction
        """

        # self.calc_baseline(threads = threads)

        # ensure the model objective is cost
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0

        # define some lists to store results and populate with the cost optimal case / baseline
        carbon_costs = [0]
        sweep_emissions = [self.cost_optimal_emissions]
        sweep_costs = [self.cost_optimal_cost]
        emissions_costs = [0]
        alignment_fraction = [0]

        # add allowable emissions parameter
        self.model.abatement_fraction = Param(initialize=1.0, mutable=True)
        self.model.allowable_emissions = Var(within = NonNegativeReals)
        @self.model.Constraint()
        def allowable_emissions_rule(m):
            return m.allowable_emissions == (self.cost_optimal_emissions - self.emissions_optimal_emissions) * (1 - m.abatement_fraction) + self.emissions_optimal_emissions
        
        @self.model.Constraint()
        def emissions_limit_rule(m):
            return m.total_flex_emissions_signal <= m.allowable_emissions

        # self.model.allowable_emissions = Param(initialize=self.cost_optimal_emissions, mutable=True)

        # while (sweep_emissions[-1] - self.emissions_optimal_emissions)/self.emissions_optimal_emissions > rel_tol:
        for a_e in np.arange(0, 1.01, 0.1):
            print(f"Calculating pareto point for abatement fraction: {a_e:.2f}")
            self.model.abatement_fraction.value = a_e

            # re-solve the model
            try:
                (self.model, results) = self.solve(self.model, threads = threads, max_iter=500)
                
                # assert results.solver.termination_condition == "optimal"
                
                total_electrical_emissions = np.sum(self.model.total_flex_emissions_signal()) 
                sweep_emissions.append(total_electrical_emissions)  

                if self.costing_type == "dam":
                    dam_cost = sum(idxparam_value(self.model.net_facility_load) * self.model.cost_signal)
                    sweep_costs.append(dam_cost)
                elif self.costing_type == "tariff":
                    self.charge_dict = costs.get_charge_dict(
                        self.startdate_dt, self.enddate_dt, self.cost_signal, resolution="1h"
                    )
                    consumption_data_dict = {"electric": idxparam_value(self.model.net_facility_load)}
                    tariff_cost = costs.calculate_cost(
                            self.charge_dict,
                            consumption_data_dict,
                            desired_utility="electric",
                            desired_charge_type=None,
                            model=None,
                            resolution="1h",
                            electric_consumption_units=u.MW
                        )[0]

                    sweep_costs.append(tariff_cost)

                else:
                    raise ValueError(f"Unknown costing type: {self.costing_type}")

                # post-process outputs
                alpha = 1 - (total_electrical_emissions - self.emissions_optimal_emissions)/(self.cost_optimal_emissions - self.emissions_optimal_emissions + 1e-8)

                # log results
                carbon_costs.append(value(self.model.cost_of_carbon)) 
                alignment_fraction.append(alpha)
            
            except:
                raise RuntimeError("Pareto point calculation failed - consider increasing the stepsize or relative tolerance")

        # convert kg to tons (metric )
        sweep_emissions = [x*0.001 for x in sweep_emissions]

        # convert $/kg to $/ton (metric)
        emissions_costs = [x*1000 for x in emissions_costs]


        # store outputs in a dataframe
        pareto_front = pd.DataFrame({"carbon_cost": carbon_costs, 
                                  "emissions": sweep_emissions, 
                                  "electricity_cost": sweep_costs, 
                                  "alignment_fraction": alignment_fraction})
        
        if savepath is not None:
            pareto_front.to_csv(savepath)
        
        return pareto_front
    

if __name__ == "__main__":
    systems = {
    "maxflex" : {
        "system_uptime": 0.0,  # minimum uptime
        "continuous_flexibility": 1.0, # full flexibility
        "pareto_stepsize": 0.01
    }, 
    "25uptime_0flex" : {
        "system_uptime": 0.25,  
        "continuous_flexibility": 0.0, 
        "pareto_stepsize": 0.05  
    },
    "50uptime_50flex" : {
        "system_uptime": 0.5,  
        "continuous_flexibility": 0.5, 
        "pareto_stepsize": 0.1 
    },
    "100uptime_75flex" : {
        "system_uptime": 1.0,  
        "continuous_flexibility": 0.75, 
        "pareto_stepsize": 0.05 
    },
    }

    region = "CAISO"
    month = 7
    year = 2023
    system_name = "maxflex"
    threads = 10 

    generate_data = False  
    basepath =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    basepath =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # get wholesale energy market + marginal emissions 
    dam = getdam(region, month, basepath)
    mef = getmef(region, month, basepath)

    # get tariff + average emissions 
    _, num_days = calendar.monthrange(year, month = month) 
    startdate_dt, enddate_dt = ms.get_start_end(month)
    tariff = gettariff(region, basepath=basepath)
    aef = getaef(region, month, basepath)   
    aef = np.tile(aef, num_days) 


    system_uptime = systems[system_name]["system_uptime"]
    continuous_flexibility = systems[system_name]["continuous_flexibility"]
    pareto_stepsize = systems[system_name]["pareto_stepsize"]

    # dam and mef pareto front 
    baseload = np.ones_like(mef)

    acc = acc_curve(
        baseload=baseload, 
        min_onsteps=max(int(len(mef) * (system_uptime)), 1), 
        flex_capacity = continuous_flexibility, 
        emissions_signal=mef, 
        emissions_type = "mef", 
        cost_signal = dam, 
        costing_type = "dam", 
        startdate_dt= startdate_dt, 
        enddate_dt=enddate_dt,
        uptime_equality=True
        )

    em_opt = acc.calc_emissions_optimal(threads = threads)
    cost_opt = acc.calc_cost_optimal(threads = threads)
    pareto_front = acc.build_pareto_front(threads = threads, stepsize=pareto_stepsize)