import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file from https://github.com/USEPA/scghg/tree/main/Meta-Analysis/output/scghgs/full_distributions/CO2
df = pd.read_csv('data/offsets/sc-CO2-meta_analysis-2030-n10000.csv')
discount_rates = df['discount_rate'].unique()
inflation = 1.18 # from July 2020 to July 2023 based on https://www.bls.gov/data/inflation_calculator.htm
all_rows = []

# Calculate percentiles for each discount rate
for i, rate in enumerate(discount_rates):
    rate_data = df[df['discount_rate'] == rate]['scghg'] * inflation
    
    # 25th, 50th, and 75th
    p25 = round(rate_data.quantile(0.25), 2)
    p50 = round(rate_data.quantile(0.50), 2) 
    p75 = round(rate_data.quantile(0.75), 2)
    
    clean_rate = float(rate.replace('% Ramsey', '')) / 100

    all_rows.extend([
        {
            'value': p50,
            'percentile': 50,
            'unit': '$/ton',
            'emission_year': 2030,
            'discount_rate': clean_rate},
        {
            'value': p25,
            'percentile': 25,
            'unit': '$/ton',
            'emission_year': 2030,
            'discount_rate': clean_rate},
        {
            'value': p75,
            'percentile': 75,
            'unit': '$/ton',
            'emission_year': 2030,
            'discount_rate': clean_rate}
    ])

scc_df = pd.DataFrame(all_rows)
scc_df.to_csv('data/offsets/scc.csv', index=False)