"""
1_load_data.py
--------------
Loads real ERCOT Historical DAM Load Zone and Hub Prices.
Source: https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er

- Reads all xlsx files from ercot_raw/ (one file per year, 12 tabs per file)
- Filters to HB_NORTH (North Texas hub benchmark price)
- Combines into a single clean hourly series
- Saves to ercot_real_prices.csv

Run first before any other script.
"""

import pandas as pd
import glob
import os
import sys

DATA_DIR  = 'ercot_raw'
OUTPUT    = 'ercot_real_prices.csv'
HUB       = 'HB_NORTH'


def load_ercot_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))

    if not files:
        print(f"ERROR: No .xlsx files found in {data_dir}/")
        print("Place your ERCOT xlsx files in the ercot_raw/ folder and try again.")
        sys.exit(1)

    dfs = []
    for f in files:
        year = f.split('DAMLZHBSPP_')[1].replace('.xlsx', '')
        xl   = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = df[df['Settlement Point'] == HUB].copy()
            dfs.append(df)
        print(f"  Loaded {year} ({len(xl.sheet_names)} months)")

    combined = pd.concat(dfs, ignore_index=True)

    # ERCOT uses hour-ending convention: "01:00" means the hour ending at 1am
    # Subtract 1 to convert to hour-starting (midnight = hour 0)
    combined['datetime'] = (
        pd.to_datetime(combined['Delivery Date'], format='%m/%d/%Y')
        + pd.to_timedelta(combined['Hour Ending'].str[:2].astype(int) - 1, unit='h')
    )

    combined = (
        combined[['datetime', 'Settlement Point Price']]
        .rename(columns={'Settlement Point Price': 'price_mwh'})
        .sort_values('datetime')
        .drop_duplicates('datetime')    # removes DST repeated hours
        .reset_index(drop=True)
    )

    # Add basic time columns (used in EDA and feature engineering)
    combined['hour']        = combined['datetime'].dt.hour
    combined['month']       = combined['datetime'].dt.month
    combined['day_of_week'] = combined['datetime'].dt.dayofweek
    combined['is_weekend']  = (combined['day_of_week'] >= 5).astype(int)
    combined['year']        = combined['datetime'].dt.year

    return combined


if __name__ == '__main__':
    print(f"Loading ERCOT {HUB} DAM prices from {DATA_DIR}/...")
    df = load_ercot_data()

    print(f"\nRows          : {len(df):,}")
    print(f"Date range    : {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Price range   : ${df['price_mwh'].min():.2f} to ${df['price_mwh'].max():,.2f}/MWh")
    print(f"Spikes >$500  : {(df['price_mwh'] > 500).sum()} hours")
    print(f"Negative hrs  : {(df['price_mwh'] < 0).sum()} hours")
    print(f"Missing hours : {df['datetime'].diff().dt.total_seconds().gt(3601).sum()}")

    uri_data = df[(df['datetime'].dt.year == 2021) & (df['datetime'].dt.month == 2)]
    print(f"Feb 2021 peak : ${uri_data['price_mwh'].max():,.2f}/MWh (URI winter storm)")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
