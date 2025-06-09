# imports
import os
import pandas as pd

def getmef(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the MEF (marginal emissions factor) for a given region and month."""
    # Load the MEF data
    mef_path = os.path.join(basepath, "data", "mef", f"{region}emissions.csv")

    # read df
    mef_data = pd.read_csv(mef_path)

    # Return the data for the specified month
    return mef_data[mef_data["month"] == month]["co2_eq_kg_per_MWh"].values

def getaef(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the AEF (average emissions factor) for a given region and month."""
    # Load the MEF data
    mef_path = os.path.join(basepath, "data", "aef", f"{region}emissions.csv")

    # read df
    mef_data = pd.read_csv(mef_path)

    # Return the data for the specified month
    return mef_data[mef_data["month"] == month]["co2_eq_kg_per_MWh"].values

def getdam(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the DAM (day-ahead market price) for a given region and month."""
    # Load the DAM data
    dam_path = os.path.join(basepath, "data", "dam", f"{region}costs.csv")

    # read df
    dam_data = pd.read_csv(dam_path)

    # Return the data for the specified month
    return dam_data[dam_data["month"] == month]["USD_per_MWh"].values

def gettariff(
    region, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the tariff for a given region."""
    # Load the tariff data
    tariff_path = os.path.join(basepath, "data", "tariff", f"{region}tariff.csv")
    
    # read df
    tariff_data = pd.read_csv(tariff_path)

    # Return the entire tariff sheet (month will be processed later)
    return tariff_data
