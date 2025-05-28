import os, gurobipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
from time import time
from warnings import warn
from electric_emission_cost import costs
from electric_emission_cost.units import u
from electric_emission_cost import utils

# Define the model
class flexloadMILP:
    def __init__(
        self,
        baseload,
        costing_type,
        costing_path,
        emissions_type,
        emissions_path,
        flex_capacity,
        rte,
        min_onsteps,
        uptime_equality=True,
        non_sheddable_fraction=0,
        cost_signal=None,
        emissions_signal=None,
        max_status_switch=None,
        horizonlength=None,
        startdate_dt=None,
        enddate_dt=None,
        electricity_costing_CWNS_No=6004010004,
        cost_col_name="LMP",
        emissions_col_name="sample_0",
        cost_of_carbon=120,
        cost_signal_units=u.USD / u.MWh,
        emissions_signal_units=u.kg / u.MWh,
        cost_of_carbon_units=u.USD / u.metric_ton,
        consumption_units=u.kW,
        numerical_tolerance=1e-8,
    ):
        """
        baseload: np.array
            The baseline power consumption of the system (MWh)
        costing_type: str
            Type of costing to use. Either "lmp", "tariff", "mef", or "aef"
        costing_path: str
            Path to the costing file. Must be either .csv or .xlsx
        emissions_type: str
            Type of emissions to use. Either "mef" or "aef"
        costing_path: str
            Path to the emissions file. Must be .csv
        flex_capacity: float [0, 1]
            The maximum capacity of the load while is it on (0-1)
        rte: float [0, 1]
            The round trip efficiency of the system (0-1)
        min_onsteps: int
            The minimum number of timesteps the load must be online
        uptime_equality: bool
            If True, the load must be online for exactly `min_onsteps` steps.
        non_sheddable_fraction: float [0, 1]
            The fraction of the load that cannot be shed.
        cost_signal: numpy.ndarray
            The cost signal if not passed in through `costing_path`
        emissions_signal: numpy.ndarray
            The emissions signal if not passed in through `emissions_path`
        max_status_switch: int, None
            The maximum number of status switches (on/off) the load can have
        electricity_costing_CWNS_No: int
            CWNS number to use if looking up tariff from Excel workbook
        horizonlength: int
            Number of timesteps in the horizon. 
            Can be determined from pricelength if not using `costing_type` of "tariff".
        cost_col_name: str
            The name of the column for costs in the cost spreadsheet
        emissions_col_name: str
            The name of the column for emissions in the emissions spreadsheet
        cost_of_carbon: float
            The price of carbon
        cost_signal_units: str or pint.Unit
            The units of the cost signal
        emissions_signal_units: str or pint.Unit
            The units of the emissions signal
        cost_of_carbon_units: str or pint.Unit
            The units of the carbon price
        consumption_units:
            The units of the baseload consumption data
        numerical_tolerance: float
            The numerical absolute tolerance used to enforce heuristic bounds and constraints. Default is 1e-8.
        """
        self.baseload = baseload
        self.costing_type = costing_type
        self.costing_path = costing_path
        self.emissions_type = emissions_type
        self.emissions_path = emissions_path
        self.flex_capacity = flex_capacity
        self.rte = rte
        self.min_onsteps = int(min_onsteps)
        self.uptime_equality = uptime_equality
        self.max_status_switch = max_status_switch
        self.non_sheddable_fraction = non_sheddable_fraction
        self.horizonlength = horizonlength
        self.startdate_dt = startdate_dt
        self.enddate_dt = enddate_dt
        self.electricity_costing_CWNS_No = electricity_costing_CWNS_No
        self.cost_col_name = cost_col_name
        self.emissions_col_name = emissions_col_name
        self.cost_of_carbon = cost_of_carbon
        self.tol = numerical_tolerance

        # load cost and emissions data
        if cost_signal is None:
            if self.costing_type == "tariff":
                fp = os.path.join(os.getcwd(), self.costing_path)
                if self.costing_path.endswith('.csv'): 
                    self.cost_signal = pd.read_csv(fp)
                elif (self.costing_path.endswith('.xlsx')):
                    self.cost_signal = pd.read_excel(fp, sheet_name=self.electricity_costing_CWNS_No)
                else: 
                    raise TypeError('File type not supported')
            elif self.costing_type == "lmp":
                fp = os.path.join(os.getcwd(), self.costing_path)
                lmp_data = pd.read_csv(fp)
                self.cost_signal = lmp_data[self.cost_col_name].values
            else:
                self.cost_signal = None
        else:
            self.cost_signal = cost_signal

        if emissions_signal is None:
            if self.emissions_type == "aef" or self.emissions_type == "mef":
                fp = os.path.join(os.getcwd(), self.emissions_path)
                emissions_data = pd.read_csv(self.emissions_path)
                self.emissions_signal = emissions_data[self.emissions_col_name]
            else:
                self.emissions_signal = None
        else:
            self.emissions_signal = emissions_signal

        if self.horizonlength is None:
            if self.costing_type == "tariff":
                self.horizonlength = int((self.enddate_dt - self.startdate_dt) / np.timedelta64(1, 'h'))
            elif self.costing_type == "lmp":
                self.horizonlength = len(self.cost_signal)
            elif self.emissions_type == "aef" or self.emissions_type == "mef":
                self.horizonlength = len(self.emissions_signal)

        # convert to default units
        if isinstance(cost_signal_units, str):
            self.cost_signal_units = u(cost_signal_units).units
        else:
            self.cost_signal_units = cost_signal_units
        if isinstance(emissions_signal_units, str):
            self.emissions_signal_units = u(emissions_signal_units).units
        else:
            self.emissions_signal_units = emissions_signal_units
        if isinstance(cost_of_carbon_units, str):
            self.cost_of_carbon_units = u(cost_of_carbon_units).units
        else:
            self.cost_of_carbon_units = cost_of_carbon_units
        if isinstance(consumption_units, str):
            self.consumption_units = u(consumption_units).units
        else:
            self.consumption_units = consumption_units

        self.baseload = (self.baseload * self.consumption_units).to(u.kW).magnitude
        if self.costing_type == "lmp":
            self.cost_signal = (self.cost_signal * self.cost_signal_units).to(u.USD / u.kWh).magnitude
        if self.emissions_type == "aef" or self.emissions_type == "mef":
            self.emissions_signal = (self.emissions_signal * self.emissions_signal_units).to(u.kg / u.kWh).magnitude
        if self.cost_of_carbon is not None:
            self.cost_of_carbon = (self.cost_of_carbon * self.cost_of_carbon_units).to(u.USD / u.kg).magnitude

        if self.max_status_switch is not None and type(self.max_status_switch) != int:
            raise ValueError("`max_status_switch` must be an integer or None")
        if self.min_onsteps > self.horizonlength:
            raise ValueError("`min_onsteps` must be less than `horizonlength`")
        if self.non_sheddable_fraction > 1 or self.non_sheddable_fraction < 0:
            raise ValueError("`non_sheddable_fraction` must be between 0 and 1")
        if len(baseload) != self.horizonlength and len(baseload) != 1:
            raise ValueError("`baseload` and `horizonlength` must have the same length")
        if flex_capacity > 1 or flex_capacity < 0:
            raise ValueError("`flex_capacity` must be between 0 and 1")
        if rte > 1 or rte < 0:
            raise ValueError("`rte` must be between 0 and 1")
        if (self.cost_signal is None) and (self.emissions_signal is None):
            raise ValueError("At least one of `cost_signal` and `emissions_signal` must be specified")


    def add_base_tariffs(self, model, resolution="1h"):
        """
        Uses the tariff data functions to add an electricity cost expression on the model
        """
        # (1) get the charge dictionary
        charge_dict = costs.get_charge_dict(
            self.startdate_dt, self.enddate_dt, self.cost_signal, resolution=resolution
        )

        # (2) set up consumption data dictionary
        consumption_data_dict = {"electric": self.baseload}

        # (3) calculate cost for objective function and modify model constraints
        model.energy_base_cost_signal, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=sum(self.baseload),
            desired_utility="electric",
            desired_charge_type="energy",
            model=None,
        )
        model.demand_base_cost_signal, _ = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=sum(self.baseload),
            desired_utility="electric",
            desired_charge_type="demand",
            model=None,
        )
        model.total_base_cost_signal = model.energy_base_cost_signal + model.demand_base_cost_signal
        return model

    def add_flex_tariffs(self, model, resolution="1h"):
        """
        Uses the tariff data functions to add an electricity cost expression on the model
        """
        # (1) get the charge dictionary
        charge_dict = costs.get_charge_dict(
            self.startdate_dt, self.enddate_dt, self.cost_signal, resolution=resolution
        )

        # (2) set up consumption data dictionary
        consumption_data_dict = {"electric": model.flexload}

        # (3) calculate cost for objective function and modify model constraints
        model.energy_flex_cost_signal, model = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=0,
            desired_utility="electric",
            desired_charge_type="energy",
            model=model,
        )
        model.demand_flex_cost_signal, model = costs.calculate_cost(
            charge_dict,
            consumption_data_dict,
            resolution=resolution,
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=0,
            desired_utility="electric",
            desired_charge_type="demand",
            model=model,
        )
        model.total_flex_cost_signal = model.demand_flex_cost_signal + model.energy_flex_cost_signal
        return model

    def build(self):
        """
        Build the model for the flexible load in pyomo.
        """
        self.t0 = time()
        model = ConcreteModel()

        # define the parameters
        model.T = Param(initialize=self.horizonlength, mutable=False)
        model.t = range(self.horizonlength)

        if self.costing_type != "tariff":
            model.pricesignal = Param(
                model.t, initialize=lambda model, t: self.pricesignal[t]
            )
        model.baseload = Param(model.t, initialize=lambda model, t: self.baseload[t])
        model.nonshedload = Param(
            model.t,
            initialize=lambda model, t: self.baseload[t] * self.non_sheddable_fraction,
        )
        model.flex_capacity = Param(initialize=self.flex_capacity)
        model.rte = Param(initialize=self.rte)
        model.min_onsteps = Param(initialize=self.min_onsteps)
        model.horizonlength = Param(initialize=self.horizonlength)
        if self.max_status_switch is not None:
            model.max_status_switch = Param(initialize=self.max_status_switch)

        # define the variables
        model.status = Var(model.t, within=Binary)
        model.onsteps = Var(bounds=(1, self.horizonlength))
        model.up = Var(model.t, within=Binary)
        model.down = Var(model.t, within=Binary)
        contload_lb = (
            (1 - self.flex_capacity)
            * sum(self.baseload)
            / (self.rte * self.horizonlength)
        )
        if self.uptime_equality:
            contload_ub = (
                (1 + self.flex_capacity)
                * sum(self.baseload)
                / (self.rte * self.min_onsteps)
            )
        else:
            contload_ub = None

        model.contload = Var(model.t, bounds=(contload_lb, contload_ub))
        model.flexload = Var(model.t, initialize=0, bounds=(0,None))

        model.max_contload = Var(bounds=(0, None))
        model.min_contload = Var(bounds=(0, None))
        model.max_contload_penalty = Param(initialize=self.tol, mutable=True)
        model.min_contload_penalty = Param(initialize=self.tol, mutable=True)

        model.flex_cost_signal = Var(model.t, within=NonNegativeReals)
        model.flex_emissions_signal = Var(model.t, within=NonNegativeReals)
        model.base_cost_signal = Var(model.t, within=NonNegativeReals)
        model.base_emissions_signal = Var(model.t, within=NonNegativeReals)
        model.net_facility_load = Var(model.t, within=NonNegativeReals)

        # define the constraints
        @model.Constraint(model.t)
        def net_flex_rule(b, t):
            return b.flexload[t] == b.contload[t] * b.status[t]

        # flexload was already created and net_facility_load lets us be consistent with the pyomo virtual battery model.
        @model.Constraint(model.t)
        def net_facility_load_rule(b, t):
            return b.net_facility_load[t] == b.flexload[t]

        @model.Constraint()
        def power_balance_rule(b):
            return sum(b.baseload[t] for t in b.t) / b.rte == sum(
                b.flexload[t] for t in b.t
            )
        
        @model.Constraint(model.t, doc="Continuous load maximum")
        def contload_max_rule(b, t):
            return b.contload[t] <= b.max_contload
        
        @model.Constraint(model.t, doc="Continuous load minimum")
        def contload_min_rule(b,t):
            return b.contload[t] >= b.min_contload

        @model.Expression(doc="Penalized continuous load range")
        def contload_range(b):
            return b.max_contload * b.max_contload_penalty - b.min_contload * b.min_contload_penalty

        @model.Constraint(doc="Number of on-steps")
        def onsteps_rule(b):
            return sum(b.status[t] for t in b.t) == b.onsteps
            
        @model.Constraint(doc="Minimum uptime constraint")
        def uptime_rule(b):
            if self.uptime_equality:
                return b.onsteps == b.min_onsteps
            else:
                return b.onsteps >= b.min_onsteps

        @model.Constraint(model.t, doc="Status update constraint")
        def status_rule(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.status[t] - b.status[t - 1] <= b.up[t] + b.down[t]

        @model.Constraint(model.t, doc="Turn down constraint")
        def down_rule(b, t):
            return b.down[t] <= 1 - b.status[t]

        @model.Constraint(model.t, doc="Turn up constraint")
        def up_rule(b, t):
            return b.up[t] <= b.status[t]

        if self.max_status_switch is not None:

            @model.Constraint()
            def shutdown_rule(b):
                return sum(b.down[t] for t in b.t) <= b.max_status_switch

            @model.Constraint()
            def startup_rule(b):
                return sum(b.up[t] for t in b.t) <= b.max_status_switch

        if self.costing_type == "tariff":
            model = self.add_base_tariffs(model)
            model = self.add_flex_tariffs(model)
        elif self.costing_type == "lmp":
            @model.Constraint(model.t, doc="Flex cost signal constraint")
            def flex_cost_signal_rule(b, t):
                return b.flex_cost_signal[t] == b.flexload[t] * b.cost_signal[t]

            @model.Constraint(model.t, doc="Base cost signal constraint")
            def base_cost_signal_rule(b, t):
                return b.base_cost_signal[t] == b.baseload[t] * b.cost_signal[t]

            @model.Expression(doc="Total cost signal for the flexible system")
            def total_flex_cost_signal(b):
                return sum(b.flex_cost_signal[t] for t in b.t)

            @model.Expression(doc="Total cost signal for the base system")
            def total_base_cost_signal(b):
                return sum(b.base_cost_signal[t] for t in b.t)
        else:
            @model.Expression(doc="Total cost signal for the flexible system")
            def total_flex_cost_signal(b):
                return 0

            @model.Expression(doc="Total cost signal for the base system")
            def total_base_cost_signal(b):
                return 0
        
        if self.emissions_type == "aef" or self.emissions_type == "mef":
            @model.Constraint(model.t, doc="Flex emissions signal constraint")
            def flex_emissions_signal_rule(b, t):
                return b.flex_emissions_signal[t] == b.flexload[t] * b.emissions_signal[t]

            @model.Constraint(model.t, doc="Base emissions signal constraint")
            def base_emissions_signal_rule(b, t):
                return b.base_emissions_signal[t] == b.baseload[t] * b.emissions_signal[t]

            @model.Expression(doc="Total emissions signal for the flexible system")
            def total_flex_emissions_signal(b):
                return sum(b.flex_emissions_signal[t] for t in b.t)

            @model.Expression(doc="Total emissions signal for the base system")
            def total_base_emissions_signal(b):
                return sum(b.base_emissions_signal[t] for t in b.t)
        else:
            @model.Expression(doc="Total emissions signal for the flexible system")
            def total_flex_emissions_signal(b):
                return 0

            @model.Expression(doc="Total emissions signal for the base system")
            def total_base_emissions_signal(b):
                return 0
        
        @model.Expression(
            doc="Percent savings from flexible operation for the cost signal of this system"
        )
        def pct_cost_savings(b):
            return (
                100
                * (b.total_base_cost_signal - b.total_flex_cost_signal)
                / b.total_base_cost_signal
            )

        @model.Expression(
            doc="Percent savings from flexible operation for the emissions signal of this system"
        )
        def pct_emissions_savings(b):
            return (
                100
                * (b.total_base_emissions_signal - b.total_flex_emissions_signal)
                / b.total_base_emissions_signal
            )

        if self.uptime_equality:
            model.objective = Objective(
                expr=model.total_flex_cost_signal + self.cost_of_carbon * model.total_flex_emissions_signal, 
                sense=minimize
            )
        else:
            # get active objectives
            model.objective = Objective(
                expr=model.total_flex_cost_signal + self.cost_of_carbon * model.total_flex_emissions_signal + model.contload_range, 
                sense=minimize
            )            

        self.model = model
        self.t1 = time()

        return model

    def calc_metrics(self, model):
        """
        Calculate the metrics of the model
        """
        self.upflex_powercapacity = max(
            [model.flexload[i]() - model.baseload[i] for i in model.t]
        ) / (sum([model.baseload[t] for t in model.t]) / model.horizonlength)
        self.discharge_capacity = sum(
            [
                model.baseload[i] - model.flexload[i]()
                for i in model.t
                if model.flexload[i]() < model.baseload[i]
            ]
        ) / (sum([model.baseload[t] for t in model.t]))

        return self.upflex_powercapacity, self.discharge_capacity

    def solve(self, tee=False, print_results=False):
        """
        Solve the model for the flexible load using gurobi
        """
        solver = SolverFactory("gurobi")
        
        self.t2 = time()

        if self.uptime_equality:
            model, results = self.solve_baseproblem(tee=tee)
        else:
            model, results = self.solve_relaxedproblem(tee=tee)

        self.t3 = time()

        if print_results:
            print(
                "Solver termination condition: ", results.solver.termination_condition
            )
            if results.solver.termination_condition == TerminationCondition.optimal:
                print("Model constructed in: ", self.t1 - self.t0)
                print("Model solved in: ", self.t3 - self.t2)
                print("Total time: ", self.t3 - self.t2 + self.t1 - self.t0)

        return self.model, results

    def solve_baseproblem(self, tee=False):
        """
        Solve the model for the flexible load using gurobi
        """
        solver = SolverFactory("gurobi")
        results = solver.solve(self.model, tee=tee)
        return self.model, results


    def solve_relaxedproblem(self, tee=False):
        """
        Solve the model for the flexible load using gurobi with bound constraints on the contload relaxed
        """
        solver = SolverFactory("gurobi")
        results = solver.solve(self.model, tee=tee)

        # check the continuous load 
        flex_capacity = self.model.flex_capacity()
        cont_load = np.array([self.model.contload[i].value for i in self.model.t if self.model.status[i].value == 1])
        cont_load_avg = np.mean(cont_load)

        # raise a warning if solution violates the bounds of the relaxed problem (within tolerance)
        if np.max(np.abs(cont_load - cont_load_avg)) - self.tol >= flex_capacity * cont_load_avg:
            print("The solution violates the bounds of flexible operation.\nResolving the problem with increased bound penalties.")
            # raise ValueError("Continuous load violated bounds of relaxed problem.")

            # TODO - think about implementing a smarter algorithm for heuristic bound enforcement
            stepsize = 1e-5

            while np.max(np.abs(cont_load - cont_load_avg)) - self.tol >= flex_capacity * cont_load_avg:

                # get the violation of the maximum continuous load
                max_contload_violation = np.max([(1+flex_capacity) * cont_load_avg - np.max(cont_load), 0])
                self.model.max_contload_penalty.value = self.model.max_contload_penalty.value + max_contload_violation * stepsize

                # get the violation of the minimum continuous load
                min_contload_violation = np.max([np.min(cont_load) - (1-flex_capacity) * cont_load_avg, 0])
                self.model.min_contload_penalty.value = self.model.min_contload_penalty.value + min_contload_violation * stepsize

                # update the model and call solve again
                results = solver.solve(self.model, tee=tee)

        return self.model, results

    def display_results(self):
        """
        Display the results of the model
        """
        # print the results
        print("\n------------------------")
        print(
            "\tBaseline: {:.2f} {}".format(
                self.model.total_base_cost_signal(), self.cost_signal_units
            )
        )
        print(
            "\tFlexible: {:.2f} {}".format(
                self.model.total_flex_cost_signal(), self.cost_signal_units
            )
        )
        print("\tCost savings: {:.2f}%".format(self.model.pct_cost_savings()))
        print("------------------------")

        flexload = np.array([self.model.flexload[t]() for t in self.model.t])
        baseload = np.array([self.model.baseload[t] for t in self.model.t])

        time = np.arange(0, self.horizonlength, 1)

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].step(time, self.cost_signal, label="Base Load", where="mid")
        ax[0].set(
            title="Price Signal",
            xlabel="Time [h]",
            ylabel="Price Signal [{}]".format(self.cost_signal_units),
            xlim=(0, self.horizonlength - 1),
        )

        ax[1].step(time, baseload, label="Base Load", where="mid")
        ax[1].step(time, flexload, label="Flexible Load", where="mid")
        ax[1].legend()
        ax[1].set(
            title="Load Profile",
            xlabel="Time [h]",
            ylabel="Power [MW]",
            ylim=(0, None),
            xlim=(0, self.horizonlength - 1),
        )
        fig.tight_layout()
        plt.show()

        return fig, ax

    def __call__(self):
        """
        Builds the model and solves it
        """
        self.build()
        self.solve(tee=False, print_results=True)
        self.calc_metrics(self.model)
        self.display_results()
        return self.model
