import os,json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analysis import emissionscost as ec


def _add_scc_and_rec(ax, regions, width, scc=True, rec=True, plot_scc_by="mean", emission_basis="mef", scc_value={"percentile": 50, "discount": 0.025}):
    """
    Overlay SCC and REC cost boxes on a given axis.
    Parameters:
    - ax: Matplotlib axis to plot on.
    - regions: List of region names corresponding to the bars on the plot.
    - width: Width of the bars in the plot.
    - scc: Boolean to indicate whether to plot SCC box.
    - rec: Boolean to indicate whether to plot REC boxes.
    - plot_scc_by: Method to plot SCC, either "mean" for 50th percentile line or "value" to use percentile/discount.
    - emission_basis: Emission basis for REC calculation, either "mef" or "aef".
    - scc_value: Dictionary with keys 'percentile' and 'discount' to specify SCC value when plot_scc_by is "value".
    """
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scc_df = pd.read_csv(os.path.join(basepath, "data", "offsets", "scc.csv"))
    rec_df = pd.read_csv(os.path.join(basepath, "data", "offsets", "rec.csv"))
    
    with open(os.path.join(basepath, "paper_figures", "code","colorscheme.json"), "r") as f:
        colors = json.load(f)
    other_colors= colors["other"]

    # define overlay parameters
    overlay_params = {
        'scc': {
            'face_color': 'black',
            'edge_color': 'black',
            'alpha': 1.0,
        },
        'rec': {
            'face_color': 'plum',
            'edge_color': 'k',
            'alpha': 1
        }
    }

        
    def _create_arrow_label(text, xy, xytext, rad=0.2, va='bottom'):
        ax.annotate(text, xy=xy, xytext=xytext,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, 
                                   connectionstyle=f'arc3,rad={rad}'),
                   ha='center', va=va, fontsize=18)
    
    if scc:  # Plot scc
        discount_rate = 0.02
        if plot_scc_by == "mean":  # Use 50th percentile to create "line"
            scc = (scc_df[scc_df['percentile'] == 50]
                   [scc_df['discount_rate'] == discount_rate]
                   ['value'].item())

            ax.hlines(scc, -100, 100, ls="--", color=other_colors["scc"])
        elif plot_scc_by == "value":  # Use min and max to create "box"
            scc = scc_df.loc[(scc_df['percentile'] == scc_value['percentile']) &
                                (scc_df['discount_rate'] == scc_value['discount']),
                                'value'].item()
            # (scc_df[scc_df['percentile'] == scc_value['percentile']]
            #        [scc_df['discount_rate'] == scc_value['discount']]
            #        ['value'].item())
            

        else:  # Use 25th and 75th percentiles
            scc_bottom = (scc_df[scc_df['percentile'] == 25]
                         [scc_df['discount_rate'] == discount_rate]
                         ['value'].item())
            scc_top = (scc_df[scc_df['percentile'] == 75]
                      [scc_df['discount_rate'] == discount_rate]
                      ['value'].item())
        
            scc_rect = Rectangle(
                (0 - width*2.5, scc_bottom),  # x, y (left edge, bottom)
                len(regions) + width*5,  # width to cover all regions
                scc_top - scc_bottom,  # height
                facecolor=other_colors['scc'],
                edgecolor=other_colors['scc'],
                alpha=0.5,
                zorder=0,  # behind the bars
            )
            ax.add_patch(scc_rect)
        
        # Arrow pointing to SCC box
        # _create_arrow_label('Social Cost\nof Carbon', 
        #                    (len(regions)/2 + 0.1, scc_top), 
        #                    (len(regions)/2 - 0.5, ax.get_ylim()[0] + 500), 
        #                    rad=-0.2)

    if rec:  # Add REC boxes
        # converting to float
        rec_df['price'] = rec_df['price'].astype(float)
        
        # Calculate average REC price for each ISO by type
        iso_rec_prices_by_type = {}
        national_data = rec_df[rec_df['iso'].str.lower() == 'national']
        national_averages = {}
        # For AEF basis, only use voluntary RECs
        rec_types_to_use = ['voluntary'] if emission_basis.lower() == "aef" else ['compliance', 'voluntary']
        for rec_type in rec_types_to_use:
            type_data = national_data[national_data['type'] == rec_type]
            if len(type_data) > 0:
                national_averages[rec_type] = type_data['price'].mean()
        
        for region in regions:
            region_lower = region.lower()
            region_data = rec_df[rec_df['iso'].str.lower() == region_lower]
            
            type_averages = {}

            if len(region_data) > 0:
                # Group by type (voluntary only for AEF) and calculate average REC prices for ISO
                rec_types_to_check = ['voluntary'] if emission_basis.lower() == "aef" else ['compliance', 'voluntary', 'srec']
                for rec_type in rec_types_to_check:
                    type_data = region_data[region_data['type'] == rec_type]
                    if len(type_data) > 0:
                        type_averages[rec_type] = type_data['price'].mean()
            
            # For regions with specific data, only use national averages for missing types
            if len(type_averages) > 0:
                for rec_type in rec_types_to_use:
                    if rec_type not in type_averages and rec_type in national_averages:
                        type_averages[rec_type] = national_averages[rec_type]
                iso_rec_prices_by_type[region] = type_averages
            else:
                # Use national average if no specific data for ISO
                if national_averages:
                    iso_rec_prices_by_type[region] = national_averages
        
        # Plot REC values for each ISO
        for region_idx, region in enumerate(regions):
                
            # Get hourly average emission factors using the helper function
            monthly_hourly_avg_emission_ton_per_mwh = ec.get_hourly_average_emission_factors(region, emission_basis)
            min_emission, max_emission = np.min(monthly_hourly_avg_emission_ton_per_mwh), np.max(monthly_hourly_avg_emission_ton_per_mwh)
            
            # Calculate max/min REC price equivalent for all REC types
            all_rec_prices = list(iso_rec_prices_by_type[region].values())
            min_rec_price_emission, max_rec_price_emission = min(all_rec_prices) / max_emission,  max(all_rec_prices) / min_emission
            
            # Rectangle spanning bars for ISO showing overall REC range
            rec_rect = Rectangle(
                (region_idx - width*2.5, min_rec_price_emission),  # x, y (left edge, bottom)
                width*5,  # width to cover all bars
                max_rec_price_emission - min_rec_price_emission,  # height
                facecolor=other_colors['rec'],
                edgecolor='k',
                alpha=0.5,
                zorder=0  # behind the bars
            )
            ax.add_patch(rec_rect)
            