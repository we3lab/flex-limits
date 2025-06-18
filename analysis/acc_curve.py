from models.flexload_milp import flexloadMILP
from pyomo.environ import value
import numpy as np
import pandas as pd 


class acc_curve(flexloadMILP):
    def __init__(self, battery_params):
        # call the parent class constructor
        super().__init__(        
            baseload=battery_params["baseload"],
            emissions_signal=battery_params["emissions_signal"],
            emissions_type=battery_params["emissions_type"],
            emissions_path=None, 
            cost_signal=battery_params["cost_signal"],
            costing_type=battery_params["cost_type"],
            # cost_signal_units=battery_params["cost_signal_units"], 
            costing_path=None, 
            flex_capacity=battery_params["flex_capacity"],
            startdate_dt=battery_params["startdate_dt"],
            enddate_dt=battery_params["enddate_dt"],
            cost_of_carbon=1.0,
            weight_of_cost=1.0, 
            rte=1.0,
            min_onsteps=battery_params["min_onsteps"],  # assuming daily cycles
            uptime_equality=True,
            )

        # define some attributes to store the parameters
        self.alpha = 0.95       # default value for alignment tolerance
        self.paretostep = 5     # default value for stepsize when sweeping pareto curve
        self.tol = 1e-5         # default value for tolerance to convergence

        # define some attributes to store the results
        self.cost_optimal_emissions = 0
        self.cost_optimal_cost = 0
        self.emissions_optimal_emissions = 0 
        self.emissions_optimal_cost = 0

        # define the base model object as an attribute of the acc_curve class
        self.model = self.build()
        return


    def calc_emissions_optimal(self):
        self.model.weight_of_cost = 0 
        self.model.cost_of_carbon = 1
        (self.model, results) = self.solve()
        
        # extract results from emissions optimal solution
        self.emissions_optimal_cost = self.model.total_flex_cost_signal()
        self.emissions_optimal_emissions = self.model.total_flex_emissions_signal()

        return self.emissions_optimal_cost, self.emissions_optimal_emissions
    
    def calc_cost_optimal(self):
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0
        (self.model, results) = self.solve()

        # extract results from cost optimal solution
        self.cost_optimal_cost = self.model.total_flex_cost_signal()
        self.cost_optimal_emissions = self.model.total_flex_emissions_signal()

        return self.cost_optimal_cost, self.cost_optimal_emissions
    
    
    def build_pareto_front(self, stepsize = 5, rel_tol = 1e-5, savepath = None):
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

        # ensure the model objective is cost
        self.model.weight_of_cost = 1 
        self.model.cost_of_carbon = 0

        # define some lists to store results and populate with the cost optimal case / baseline
        carbon_costs = [0]
        sweep_emissions = [self.cost_optimal_emissions]
        sweep_costs = [self.cost_optimal_cost]
        emissions_costs = [0]
        alignment_fraction = [0]

        # loop through the pareto curve until the emissions optimal solution is (nearly) reached
        print(self.cost_optimal_emissions, self.emissions_optimal_emissions)

        while (sweep_emissions[-1] - self.emissions_optimal_emissions)/self.emissions_optimal_emissions > rel_tol:
            # step up the carbon cost
            self.model.cost_of_carbon = value(self.model.cost_of_carbon) + stepsize
            
            # re-solve the model
            (self.model, results) = self.solve(self.model)
            total_electrical_emissions = np.sum(self.model.total_flex_emissions_signal()) 

            # post-process outputs
            alpha = 1 - (total_electrical_emissions - self.emissions_optimal_emissions)/(self.cost_optimal_emissions - self.emissions_optimal_emissions)

            # log results
            sweep_emissions.append(total_electrical_emissions)  
            sweep_costs.append(self.model.total_flex_cost_signal())
            carbon_costs.append(value(self.model.cost_of_carbon)) 
            emissions_costs.append(self.model.total_flex_emissions_signal()*value(self.model.cost_of_carbon))
            alignment_fraction.append(alpha)

        #convert kg to tons (metric )
        sweep_emissions = [x*0.001 for x in sweep_emissions]

        # convert $/kg to $/ton (metric)
        emissions_costs = [x*1000 for x in emissions_costs]


        # store outputs in a dataframe
        pareto_front = pd.DataFrame({"carbon_cost": carbon_costs, 
                                  "emissions": sweep_emissions, 
                                  "emissions_cost": emissions_costs, 
                                  "electricity_cost": sweep_costs, 
                                  "alignment_fraction": alignment_fraction})

        
        if savepath is not None:
            pareto_front.to_csv(savepath)
        
        return pareto_front
    


