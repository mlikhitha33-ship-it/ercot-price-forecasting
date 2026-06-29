"""
2_eda.py
--------
Exploratory Data Analysis for ERCOT HB_NORTH day-ahead prices.

Reads  : ercot_real_prices.csv  (output of 1_load_data.py)
Outputs: ercot_eda.png          (4-panel chart)
Prints : stationarity test result (ADF test)

Run after 1_load_data.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

INPUT = 'ercot_real_prices.csv'


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'ERCOT HB_NORTH Day-Ahead Market Prices (2019-2026)',
        fontsize=14, fontweight='bold'
    )

    # ── Plot 1: Full price series ─────────────────────────────────────────────
    # Clip at $500 so the Feb 2021 URI spike ($8,999) doesn't squash everything
    axes[0, 0].plot(
        df['datetime'], df['price_mwh'].clip(upper=500),
        linewidth=0.4, color='#2A7F8F', alpha=0.8
    )
    axes[0, 0].set_title('Full Price Series (clipped at $500 — URI spike hit $8,999)')
    axes[0, 0].set_ylabel('Price ($/MWh)')
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=30)

    # ── Plot 2: Median price by hour of day ──────────────────────────────────
    # Use median not mean — mean is distorted by URI spike
    hourly_avg = df.groupby('hour')['price_mwh'].median()
    axes[0, 1].bar(
        hourly_avg.index, hourly_avg.values,
        color='#2A7F8F', alpha=0.8, edgecolor='white'
    )
    axes[0, 1].set_title('Median Price by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Median Price ($/MWh)')
    axes[0, 1].set_xticks(range(0, 24, 2))

    # ── Plot 3: Median price by month ─────────────────────────────────────────
    monthly_avg = df.groupby('month')['price_mwh'].median()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['#d94f3d' if m in [6, 7, 8, 9] else '#2A7F8F' for m in range(1, 13)]
    axes[1, 0].bar(
        range(1, 13), monthly_avg.values,
        color=colors, alpha=0.85, edgecolor='white'
    )
    axes[1, 0].set_title('Median Price by Month (Red = TX Summer Peak)')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names)
    axes[1, 0].set_ylabel('Median Price ($/MWh)')

    # ── Plot 4: Price distribution ────────────────────────────────────────────
    # Clip at $200 — URI spike would make the histogram unreadable
    axes[1, 1].hist(
        df['price_mwh'].clip(-50, 200),
        bins=100, color='#2A7F8F', alpha=0.8, edgecolor='white'
    )
    axes[1, 1].axvline(
        df['price_mwh'].median(), color='#d94f3d', linestyle='--',
        label=f"Median: ${df['price_mwh'].median():.1f}"
    )
    axes[1, 1].set_title('Price Distribution (clipped at $200 for readability)')
    axes[1, 1].set_xlabel('Price ($/MWh)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('ercot_eda.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved ercot_eda.png")


def run_stationarity_test(df: pd.DataFrame) -> None:
    print("\nStationarity Test (Augmented Dickey-Fuller)")
    print("-" * 45)

    # Use daily median — more stable than hourly for ADF
    # SARIMA requires stationarity; LSTM does not
    daily = df.set_index('datetime')['price_mwh'].resample('D').median()

    result = adfuller(daily.dropna())
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.6f}")
    print(f"Critical (5%) : {result[4]['5%']:.4f}")
    print()

    if result[1] < 0.05:
        print("Result: STATIONARY (p < 0.05)")
        print("SARIMA can be applied without differencing.")
    else:
        print("Result: NON-STATIONARY (p > 0.05)")
        print("Consider first-differencing before applying SARIMA.")


if __name__ == '__main__':
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])
    print(f"Rows: {len(df):,} | {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    print("\nGenerating EDA plots...")
    plot_eda(df)

    run_stationarity_test(df)
