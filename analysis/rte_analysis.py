import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.pricesignal import getmef, getdam
from models.flexload_milp import flexloadMILP

def savings_rte(uptime, power_capacity, rte, region, month, signal="cost"):
    """
    Calculate the maximum savings as a percentage for a given region, month, and signal type (cost or emissions)
    using a specified round-trip efficiency (RTE).
    Parameters
    ----------
    uptime : float
        The fraction of time the flexible load is operational (between 0 and 1).
    power_capacity : float
        The power capacity of the flexible load in MW.
    rte : float
        The round-trip efficiency of the flexible load (between 0 and 1)
    region : str
        The name of the ISO to get the data for.
    month : int
        The month to get the data for.
    signal : str
        The type of signal to use, either "cost" or "emissions".
    """
    
    if signal=="cost":
        data = getdam(region, month)
    elif signal=="emissions":
        data = getmef(region, month)
    else:
        raise ValueError("Signal must be either 'cost' or 'emissions'")

    
    flex = flexloadMILP(
        baseload=np.ones_like(data),
        cost_signal=data,
        costing_type="dam",
        costing_path=None,
        emissions_type=None,
        emissions_path=None,
        flex_capacity=power_capacity,
        rte=rte,
        min_onsteps=int(len(data) * uptime),  # assuming daily cycles
        uptime_equality=True,
    )

    flex.build()
    flex.solve()

    return flex.model.pct_cost_savings()
