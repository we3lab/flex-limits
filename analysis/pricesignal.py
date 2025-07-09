# imports
import os
import pandas as pd

def getmef(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the MEF (marginal emissions factor) for a given region and month.
    
    Parameters
    ----------
    region : str
        The name of the ISO to get the MEF for.

    month : int
        The month to get the MEF for.

    basepath : str
        The path to the root where the data folder is located. 
        Default is the root of this repository.
    
    Returns
    -------
    numpy.array
        Array of MEFs for given `region` and `month`
    """
    # Load the MEF data
    mef_path = os.path.join(basepath, "data", "mef", f"{region}emissions.csv")

    # read df
    mef_data = pd.read_csv(mef_path)

    # Return the data for the specified month
    return mef_data[mef_data["month"] == month]["co2_eq_kg_per_MWh"].values

def getaef(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the AEF (average emissions factor) for a given region and month.
    
    Parameters
    ----------
    region : str
        The name of the ISO to get the AEF for.

    month : int
        The month to get the AEF for.

    basepath : str
        The path to the root where the data folder is located. 
        Default is the root of this repository.
    
    Returns
    -------
    numpy.array
        Array of AEFs for given `region` and `month`
    """
    # Load the MEF data
    mef_path = os.path.join(basepath, "data", "aef", f"{region}emissions.csv")

    # read df
    mef_data = pd.read_csv(mef_path)

    # Return the data for the specified month
    return mef_data[mef_data["month"] == month]["co2_eq_kg_per_MWh"].values

def getdam(
    region, month, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get the DAM (day-ahead market price) for a given region and month.
    
    Parameters
    ----------
    region : str
        The name of the ISO to get the DAM for.

    month : int
        The month to get the DAM for.

    basepath : str
        The path to the root where the data folder is located. 
        Default is the root of this repository.
    
    Returns
    -------
    numpy.array
        Array of DAM prices for given `region` and `month`
    """
    # Load the DAM data
    dam_path = os.path.join(basepath, "data", "dam", f"{region}costs.csv")

    # read df
    dam_data = pd.read_csv(dam_path)

    # Return the data for the specified month
    return dam_data[dam_data["month"] == month]["USD_per_MWh"].values

def gettariff(
    region, full_list=False, return_ids = False, basepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
):
    """Get a single tariff or list of tariffs for a given region.
    
    Parameters
    ----------
    region : str
        The name of the ISO to get the tariff for.

    full_list : bool
        Whether to get a single sample tariff or the full list of tariffs for `region`.
        False by default, meaning only a single tariff is returned.

    return_ids: bool 
        Whether to return the tariff ids in addition to the list of tariff data. 
        This option is not available for a single tariff. False by default, meaning only the tariff data is returned. 

    basepath : str
        The path to the root where the data folder is located. 
        Default is the root of this repository.
    
    Returns
    -------
    pandas.DataFrame or list of pandas.DataFrame
        Data for a single tariff or list of tariff sheets for the `region` 
        depending on value of `full_list` parameter
    """
    tariff_base = os.path.join(basepath, "data", "tariff")
    if full_list:
        tariff_list = []
        tariff_id_list  = []
        metadata_df = pd.read_csv(os.path.join(tariff_base, "metadata.csv"))
        for tariff_id in metadata_df["label"][metadata_df["ISO"] == region]:
            try:
                tariff = pd.read_csv(os.path.join(tariff_base, "bundled", tariff_id + ".csv"))
                tariff_list.append(tariff)
                tariff_id_list.append(tariff_id)
            except FileNotFoundError:
                print(f"Tariff {tariff_id} not found in {os.path.join(tariff_base, 'bundled')}")

        if return_ids:         
            return tariff_list, tariff_id_list
        else: 
            return tariff_list
    else:
        # Load the tariff data
        tariff_path = os.path.join(tariff_base, f"{region}tariff.csv")
        
        # read df
        tariff_data = pd.read_csv(tariff_path)

        # Return the entire tariff sheet (month will be processed later)
        return tariff_data
