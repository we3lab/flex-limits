# imports
import os
import pandas as pd

# make the basepath the root folder of the repo
basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def getmef(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """
    Get the MEF (Marginal emissions factor) for a given region and month.
    """
    # Load the MEF data
    mef_path = os.path.join(basepath, "data", "mef", f"{region}emissions.csv")

    # read df
    mef_data = pd.read_csv(mef_path)

    # Return the data for the specified month
    return mef_data[mef_data["month"] == month]["co2_eq_kg_per_MWh"].values


def getlmp(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """
    Get the LMP (Locational Marginal Price) for a given region and month.
    """
    # Load the LMP data
    lmp_path = os.path.join(basepath, "data", "lmp", f"{region}costs.csv")

    # read df
    lmp_data = pd.read_csv(lmp_path)

    # Return the data for the specified month
    return lmp_data[lmp_data["month"] == month]["USD_per_MWh"].values
