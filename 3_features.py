"""
3_features.py
-------------
Feature engineering for ERCOT electricity price forecasting.

Reads  : ercot_real_prices.csv   (output of 1_load_data.py)
Outputs: ercot_features.csv      (model-ready feature matrix)

Features built:
- Cyclical time encoding (sin/cos for hour, month, day-of-week)
- Lagged price values (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)
- Rolling statistics (24h and 7-day mean/std)
- Price winsorized at 99.9th percentile (handles URI spike for scaling)

The winsorization cap is computed from training period prices
(2019-2023) only, then applied to the full dataset. Computing it
from the full dataset would let validation/test period prices
(2024-2026) influence a transform applied to the training set,
which is a form of data leakage.

Run after 1_load_data.py.
"""

import pandas as pd
import numpy as np

INPUT  = 'ercot_real_prices.csv'
OUTPUT = 'ercot_features.csv'

# Features the LSTM will use — order matters (price_mwh must be index 0)
FEATURE_COLS = [
    'price_mwh',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
    'is_weekend',
    'price_lag_1h', 'price_lag_2h', 'price_lag_3h', 'price_lag_6h',
    'price_lag_12h', 'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
    'price_roll_24h_mean', 'price_roll_24h_std',
    'price_roll_7d_mean',  'price_roll_7d_std',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Winsorize price at 99.9th percentile, training data only ─────────────
    # The Feb 2021 URI spike ($8,999/MWh) would dominate MinMaxScaler,
    # compressing all normal-range prices into a tiny band near zero.
    # The cap is computed from training period prices (2019-2023) only.
    # Computing it on the full dataset would let validation/test data
    # (2024-2026) influence a value used to transform the training set,
    # which is a form of data leakage.
    train_mask = df['datetime'] < '2024-01-01'
    p999 = df.loc[train_mask, 'price_mwh'].quantile(0.999)
    print(f"99.9th percentile price (training data only) : ${p999:.2f}/MWh")
    print(f"Prices above this cap (full dataset)          : {(df['price_mwh'] > p999).sum()} hours")
    df['price_mwh_raw'] = df['price_mwh']
    df['price_mwh'] = df['price_mwh'].clip(upper=p999)

    # ── Cyclical time encoding ────────────────────────────────────────────────
    # Why sin/cos instead of raw numbers or one-hot?
    # Hour 23 and hour 0 are adjacent in real life but far apart numerically.
    # Sin/cos encoding wraps the cycle so the model sees them as neighbors.
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ── Lagged price features ─────────────────────────────────────────────────
    # Same hour yesterday (24h) and same hour last week (168h) are strong
    # predictors in electricity markets — weekly patterns are very consistent.
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'price_lag_{lag}h'] = df['price_mwh'].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    # shift(1) on all rolling features prevents data leakage —
    # we only use information available BEFORE the current hour.
    df['price_roll_24h_mean'] = df['price_mwh'].shift(1).rolling(24).mean()
    df['price_roll_24h_std']  = df['price_mwh'].shift(1).rolling(24).std()
    df['price_roll_7d_mean']  = df['price_mwh'].shift(1).rolling(168).mean()
    df['price_roll_7d_std']   = df['price_mwh'].shift(1).rolling(168).std()

    # Drop rows with NaN (from lags and rolling windows)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    return df


if __name__ == '__main__':
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])
    print(f"Rows before features: {len(df):,}")

    print("\nBuilding features...")
    df_feat = build_features(df)

    print(f"\nRows after dropping NaN: {len(df_feat):,}")
    print(f"Features built: {len(FEATURE_COLS)}")
    print("\nFeature list:")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")

    # Save datetime + all feature cols + raw unwinsorized price for reference
    cols_to_save = ['datetime'] + FEATURE_COLS + ['price_mwh_raw'] + \
                   [c for c in ['hour', 'month', 'day_of_week', 'year']
                    if c in df_feat.columns and c not in FEATURE_COLS]
    df_feat[cols_to_save].to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
