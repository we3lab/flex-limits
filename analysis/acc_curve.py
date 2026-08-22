from models.flexload_milp import flexloadMILP, idxparam_value
from pyomo.environ import Param, Constraint, Var, NonNegativeReals
from pyomo.opt import TerminationCondition
import numpy as np
import pandas as pd
import os
import calendar
from analysis.pricesignal import getmef, getaef, getdam, gettariff
import analysis.maxsavings as ms
from eeco import costs
from eeco.units import u


def get_mac(df, tol=1e-9, cost_tol=1e-7):
    """
    **Description**:

    Calculate the marginal abatement cost of each pareto point relative to the cost optimal
    point of the same sweep.

    The cost optimal point is the reference, so it has no abatement cost of its own and is
    excluded. Points that abate no emissions relative to the reference are also excluded -
    the abatement cost is undefined there, not zero.

    **Parameters**:

        df (DataFrame): a pareto front as written by acc_curve.build_pareto_front
        tol (float - optional): the emissions abatement below which a point is treated as
            abating nothing relative to the cost optimal point
        cost_tol (float - optional): cost differences smaller than this fraction of the
            reference cost are treated as exactly zero. Tariff costs are piecewise flat in
            the load schedule, so a whole band of schedules costs the same to the cent and
            the differences that survive are float noise, not cheap abatement.

    **Returns**:

        shadow_cost (list): the abatement cost of each retained point in $/ton
        emissions (list): the total emissions of each retained point, in the same units as
            the input dataframe
    """
    # the cost optimal point is the one that abates the least, i.e. emits the most
    cost_optimal = df.loc[df["emissions"].idxmax()]

    shadow_cost = []
    emissions = []
    cost_scale = abs(cost_optimal["electricity_cost"])
    for index, row in df.drop(index=cost_optimal.name).iterrows():
        delta_cost = row["electricity_cost"] - cost_optimal["electricity_cost"]
        delta_emissions = cost_optimal["emissions"] - row["emissions"]

        # this schedule sits in the same flat cost band as the reference: abatement is free
        if abs(delta_cost) <= cost_tol * cost_scale:
            delta_cost = 0.0

        # no abatement relative to the reference means there is no abatement cost to report
        if delta_emissions <= tol:
            continue

        shadow_cost.append(delta_cost / delta_emissions)
        emissions.append(row["emissions"])

    return shadow_cost, emissions


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

    def calc_emissions_optimal(self, threads = 10, resolution="1h", mipgap = 1e-6):
        self.model.weight_of_cost = 0 
        self.model.cost_of_carbon = 1
        (self.model, results) = self.solve(threads = threads, max_iter=100, descent_stepsize=0.1, mipgap = mipgap)
        
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
    
    def calc_cost_optimal(self, threads = 10, resolution="1h", mipgap = 1e-6):
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0
        (self.model, results) = self.solve(threads = threads, max_iter=100, descent_stepsize=0.1, mipgap = mipgap)

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

    def build_pareto_front(self, stepsize=5, rel_tol = 1e-5, threads = 10, savepath = None, mipgap = None, gap_fraction = 1e-3):
        """
        **Description**:

        Function sweeps the pareto curve to find the relationship between the carbon cost and the fraction of potential emissions savings.

        **Parameters**:

            stepsize (float - optional): the resolution of the pareto curve sweep in units of carbon cost
            rel_tol (float - optional): the relative slack added to the emissions cap. At
                abatement_fraction = 1 the cap is the emissions optimal value itself, which
                is infeasible by a floating point hair because that value was recomputed in
                python from the solution rather than read back from the solver.
            savepath (str - optional): the path to save the results to a csv file
            mipgap (float - optional): the relative MIP gap for each sweep solve. Defaults to
                None, which derives one from the cost span of the front - see gap_fraction.
                Pass a value to override.
            gap_fraction (float - optional): the fraction of the front's own cost span that
                the solver is allowed as absolute optimality gap. A *relative* gap is the
                wrong tolerance here: the front spans only the cost difference between the
                cost optimal and emissions optimal solutions, which can be a tiny fraction
                of the objective. On CAISO July wholesale the whole front spans $2.30 of a
                ~$934 objective, so the solver default of 1e-2 (+/- $9.3) swallows it whole
                and the sweep collapses onto the emissions optimal point.

        **Returns**:

            pareto_front (DataFrame): a dataframe containing the abatement fraction requested of each sweep point, the total electrical emissions, the electricity cost, and the realised alignment fraction
        """

        # self.calc_baseline(threads = threads)

        # ensure the model objective is cost
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0

        if mipgap is None:
            # size the gap against the cost span of the front rather than the objective
            cost_span = abs(self.cost_optimal_cost - self.emissions_optimal_cost)
            objective_scale = max(abs(self.cost_optimal_cost), abs(self.emissions_optimal_cost))
            mipgap = float(np.clip(gap_fraction * cost_span / objective_scale, 1e-9, 1e-2))
            print(
                f"Front spans {cost_span:.4g} of a {objective_scale:.4g} objective "
                f"-> using mipgap {mipgap:.3g}"
            )

        # seed the results with the cost optimal case. this is the reference every abatement
        # cost is measured against, so it is taken from calc_cost_optimal rather than from a
        # sweep solve - under a tariff many schedules tie on cost exactly, and an
        # abatement_fraction = 0 solve is free to return any one of them. the sweep then runs
        # abatement_fraction = 0 as well, so the first two rows are the reference and the
        # sweep's own read on it: identical where the problem is well determined, and a
        # visible warning of cost degeneracy where they differ.
        sweep_emissions = [self.cost_optimal_emissions]
        sweep_costs = [self.cost_optimal_cost]
        alignment_fraction = [0]
        abatement_fractions = [0]

        # add allowable emissions parameter
        self.model.abatement_fraction = Param(initialize=1.0, mutable=True)
        self.model.allowable_emissions = Var(within = NonNegativeReals)
        emissions_cap_slack = rel_tol * abs(self.emissions_optimal_emissions)

        @self.model.Constraint()
        def allowable_emissions_rule(m):
            return m.allowable_emissions == (self.cost_optimal_emissions - self.emissions_optimal_emissions) * (1 - m.abatement_fraction) + self.emissions_optimal_emissions + emissions_cap_slack
        
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
                (self.model, results) = self.solve(threads = threads, max_iter=500, mipgap = mipgap)

                assert results.solver.termination_condition == TerminationCondition.optimal, (
                    f"Solver terminated with {results.solver.termination_condition}, not optimal"
                )

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
                alignment_fraction.append(alpha)
                abatement_fractions.append(a_e)

            except Exception as e:
                raise RuntimeError(
                    f"Pareto point calculation failed at abatement fraction {a_e:.2f}"
                ) from e

        # convert kg to tons (metric )
        sweep_emissions = [x*0.001 for x in sweep_emissions]

        # store outputs in a dataframe
        pareto_front = pd.DataFrame({"abatement_fraction": abatement_fractions,
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