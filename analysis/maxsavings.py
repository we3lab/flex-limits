# imports
import os
import analysis.pricesignal as ps
from models.flexload_milp import flexloadMILP


region="CAISO"
month=1

def max_mef_savings(data, system_uptime, continuous_flex, baseload):

    flex = flexloadMILP(baseload=baseload, 
                        emissions_signal=data,
                        costing_type=None,
                        costing_path=None,
                        emissions_type="mef",
                        emissions_path=None,
                        flex_capacity=continuous_flex, 
                        rte=1.0,
                        min_onsteps=max(int(len(data)*(system_uptime)),1),  # assuming daily cycles
                        uptime_equality=True)
    
    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()

def max_aef_savings(data, system_uptime, continuous_flex, baseload):

    flex = flexloadMILP(baseload=baseload, 
                        emissions_signal=data,
                        costing_type=None,
                        costing_path=None,
                        emissions_type="aef",
                        emissions_path=None,
                        flex_capacity=continuous_flex, 
                        rte=1.0,
                        min_onsteps=max(int(len(data)*(system_uptime)),1),  # assuming daily cycles
                        uptime_equality=True)
    
    flex.build()
    flex.solve()

    return flex.model.pct_emissions_savings()

def max_lmp_savings(data, system_uptime, continuous_flex, baseload):

    flex = flexloadMILP(baseload=baseload, 
                        cost_signal=data,
                        costing_type="lmp",
                        costing_path=None,
                        emissions_type=None,
                        emissions_path=None,
                        flex_capacity=continuous_flex, 
                        rte=1.0,
                        min_onsteps=max(int(len(data)*(system_uptime)),1),  # assuming daily cycles
                        uptime_equality=True)
    
    flex.build()
    flex.solve()

    return flex.model.pct_cost_savings()


# TODO - implement tariff savings for a single case