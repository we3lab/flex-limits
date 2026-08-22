# imports
import numpy as np
from eeco import costs
from eeco.units import u

# charge types as `eeco` names them in the charge-dict key
DEMAND = "demand"
ENERGY = "energy"
CUSTOMER = "customer"

# the `period` label `eeco` gives a demand charge that applies over the whole billing
# window rather than to a time-of-use window. A non-coincident charge like this is billed
# on the monthly peak whenever it occurs, so no amount of load shifting avoids it.
NONCOINCIDENT = "flat"

# order of the structural features returned by `tariff_charge_features`, and the order they
# are clustered and plotted in
FEATURE_COLS = [
    "dem_share",
    "dem_to_eng",
    "n_demand",
    "n_flat_dem",
    "n_tou_dem",
    "dem_rate_max",
    "n_energy",
    "eng_spread",
    "n_charges",
    "n_tiers",
]


def window_length(startdate_dt, enddate_dt):
    """Number of hourly timesteps in a billing window.

    Parameters
    ----------
    startdate_dt : numpy.datetime64
        First timestep of the billing window.

    enddate_dt : numpy.datetime64
        One timestep past the last, as returned by `analysis.maxsavings.get_start_end`.

    Returns
    -------
    int
        The number of hours spanned by the window
    """
    return int((enddate_dt - startdate_dt) / np.timedelta64(1, "h"))


def parse_key(key):
    """Split an `eeco` charge-dict key into its six fields.

    The key format is `utility_chargetype_name_effstart_effend_chargelimit`, built by
    `eeco.costs.get_charge_df` and documented at `eeco.costs.get_unique_row_name`.

    Parameters
    ----------
    key : str
        A key of the dict returned by `eeco.costs.get_charge_dict`.

    Returns
    -------
    tuple
        `(utility, charge_type, name, eff_start, eff_end, charge_limit)`, with
        `charge_limit` as an int
    """
    utility, charge_type, name, eff_start, eff_end, limit = key.split("_")
    return utility, charge_type, name, eff_start, eff_end, int(limit)


def baseline_bill(
    charge_dict, baseload, resolution="1h",
):
    """Itemize the bill an inflexible load would pay under one tariff.

    This is a cost *evaluation*, not an optimization: it is the same
    `eeco.costs.calculate_cost` call that `analysis.maxsavings.max_tariff_savings` uses for
    its `total_base_cost`, so the number is directly comparable to the
    `cost_optimal_cost_usd` already stored in `paper_figures/processed_data/`.

    Splitting the bill by charge type is what makes "how demand-charge-heavy is this
    tariff" an exact quantity rather than a heuristic over the raw $/kW rates: it prices
    the tariff's own tier and time-of-use structure against a known load.

    Parameters
    ----------
    charge_dict : dict
        Charge arrays from `eeco.costs.get_charge_dict`.

    baseload : numpy.array
        The load profile to price, in MW.

    resolution : str
        Timeseries resolution of `baseload`. Default is "1h".

    Returns
    -------
    dict
        Bill in USD under keys "total", "demand", "energy" and "customer"
    """
    kwargs = dict(
        resolution=resolution,
        desired_utility="electric",
        model=None,
        electric_consumption_units=u.MW,
    )
    bill = {}
    for label, charge_type in (
        ("total", None),
        ("demand", DEMAND),
        ("energy", ENERGY),
        ("customer", CUSTOMER),
    ):
        cost, _ = costs.calculate_cost(
            charge_dict,
            {"electric": baseload},
            desired_charge_type=charge_type,
            **kwargs,
        )
        bill[label] = float(cost)

    return bill


def energy_rate_envelope(charge_dict, horizonlength):
    """Hourly energy price seen by the first consumption tier.

    Time-of-use energy periods are separate entries in `charge_dict`, each zero outside its
    own hours, so the price actually faced at a given hour is the elementwise maximum over
    the base-tier energy arrays. Higher tiers are excluded because which of them binds
    depends on cumulative consumption, and the point of this envelope is the price
    *variation* a load can shift against.

    Parameters
    ----------
    charge_dict : dict
        Charge arrays from `eeco.costs.get_charge_dict`.

    horizonlength : int
        Number of timesteps in the billing window.

    Returns
    -------
    numpy.array
        Array of `horizonlength` energy rates in $/kWh
    """
    envelope = np.zeros(horizonlength)
    for key, array in charge_dict.items():
        _, charge_type, _, _, _, limit = parse_key(key)
        if charge_type == ENERGY and limit == 0:
            envelope = np.maximum(envelope, np.asarray(array).ravel()[:horizonlength])

    return envelope


def tariff_charge_features(charge_dict, baseload, resolution="1h"):
    """Structural features of one tariff over one billing window.

    Features are read off the charge dict rather than the raw tariff CSV so that month
    applicability, hour/weekday windows and tier expansion are all resolved by `eeco`
    first. A tariff therefore has different features in January and July whenever its rates
    are seasonal, which is why the unit of analysis is a (tariff, month) pair.

    Charge *counts* collapse tiers, counting distinct period names, because a tiered charge
    is one charge the operator faces, not several.

    Parameters
    ----------
    charge_dict : dict
        Charge arrays from `eeco.costs.get_charge_dict`.

    baseload : numpy.array
        The inflexible reference load used to price the bill, in MW.

    resolution : str
        Timeseries resolution of `baseload`. Default is "1h".

    Returns
    -------
    dict
        The features named in `FEATURE_COLS`, plus the itemized reference bill under
        "base_cost", "base_demand_cost", "base_energy_cost" and "base_customer_cost"
    """
    parsed = [parse_key(key) for key in charge_dict]
    demand_keys = [p for p in parsed if p[1] == DEMAND]
    energy_keys = [p for p in parsed if p[1] == ENERGY]

    bill = baseline_bill(charge_dict, baseload, resolution=resolution)
    envelope = energy_rate_envelope(charge_dict, len(baseload))

    # spread of the energy price over the window, normalized by its own mean so that it
    # measures how much shape there is to arbitrage rather than how expensive the tariff
    # is. Zero-price hours are excluded from the minimum: they are hours the charge simply
    # does not cover, not a rate of zero.
    priced = envelope[envelope > 0]
    if priced.size and envelope.mean() > 0:
        eng_spread = float((envelope.max() - priced.min()) / envelope.mean())
    else:
        eng_spread = 0.0

    demand_rates = [
        float(np.max(charge_dict["_".join(str(f) for f in p)])) for p in demand_keys
    ]

    return {
        # magnitude of the demand charge relative to the rest of the bill
        "dem_share": bill["demand"] / bill["total"] if bill["total"] else 0.0,
        "dem_to_eng": bill["demand"] / bill["energy"] if bill["energy"] else np.nan,
        # how many charges, and of what kind
        "n_demand": len({p[2] for p in demand_keys}),
        "n_flat_dem": len({p[2] for p in demand_keys if p[2] == NONCOINCIDENT}),
        "n_tou_dem": len({p[2] for p in demand_keys if p[2] != NONCOINCIDENT}),
        "dem_rate_max": max(demand_rates, default=0.0),
        "n_energy": len({p[2] for p in energy_keys}),
        "eng_spread": eng_spread,
        "n_charges": len(parsed),
        "n_tiers": sum(1 for p in parsed if p[5] > 0),
        # the reference bill itself, kept so the cached features are self-contained
        "base_cost": bill["total"],
        "base_demand_cost": bill["demand"],
        "base_energy_cost": bill["energy"],
        "base_customer_cost": bill["customer"],
    }


def features_for_tariff(tariff_data, startdate_dt, enddate_dt, resolution="1h"):
    """Structural features of one tariff, building the charge dict from a rate sheet.

    Convenience wrapper over `tariff_charge_features` for callers holding a tariff sheet as
    loaded by `analysis.pricesignal.gettariff`. The reference load is a flat 1 MW profile,
    matching the baseload the shadow-cost results were generated against.

    Parameters
    ----------
    tariff_data : pandas.DataFrame
        A tariff rate sheet, e.g. one element of `gettariff(region, full_list=True)`.

    startdate_dt : numpy.datetime64
        First timestep of the billing window.

    enddate_dt : numpy.datetime64
        One timestep past the last.

    resolution : str
        Timeseries resolution to price at. Default is "1h".

    Returns
    -------
    dict
        As `tariff_charge_features`
    """
    charge_dict = costs.get_charge_dict(
        startdate_dt, enddate_dt, tariff_data, resolution=resolution
    )
    baseload = np.ones(window_length(startdate_dt, enddate_dt))

    return tariff_charge_features(charge_dict, baseload, resolution=resolution)
