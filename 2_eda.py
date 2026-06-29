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
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    fig.suptitle(
        'ERCOT HB_NORTH Day-Ahead Market Prices (2019-2026)',
        fontsize=14, fontweight='bold'
    )

    # ── Plot 1: Median price by hour of day ──────────────────────────────────
    hourly_avg = df.groupby('hour')['price_mwh'].median()
    axes[0, 0].bar(
        hourly_avg.index, hourly_avg.values,
        color='#2A7F8F', alpha=0.8, edgecolor='white'
    )
    axes[0, 0].set_title('Median Price by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Median Price ($/MWh)')
    axes[0, 0].set_xticks(range(0, 24, 2))

    # ── Plot 2: Median price by month ─────────────────────────────────────────
    monthly_avg = df.groupby('month')['price_mwh'].median()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['#d94f3d' if m in [6, 7, 8, 9] else '#2A7F8F' for m in range(1, 13)]
    axes[0, 1].bar(
        range(1, 13), monthly_avg.values,
        color=colors, alpha=0.85, edgecolor='white'
    )
    axes[0, 1].set_title('Median Price by Month (Red = TX Summer Peak)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(month_names, rotation=30, ha='right')
    axes[0, 1].set_ylabel('Median Price ($/MWh)')

    # ── Plot 3: Price distribution ────────────────────────────────────────────
    plot_data = df['price_mwh'].clip(-10, 150)
    axes[1, 0].hist(
        plot_data,
        bins=150, color='#2A7F8F', alpha=0.8, edgecolor='none'
    )
    axes[1, 0].axvline(
        df['price_mwh'].median(), color='#d94f3d', linestyle='--', linewidth=1.5,
        label=f"Median: ${df['price_mwh'].median():.1f}/MWh"
    )

    pct_below_50  = (df['price_mwh'] < 50).mean() * 100
    pct_above_100 = (df['price_mwh'] > 100).mean() * 100
    pct_negative  = (df['price_mwh'] < 0).mean() * 100

    axes[1, 0].text(
        0.97, 0.95,
        f"{pct_below_50:.1f}% of hours below $50\n"
        f"{pct_above_100:.1f}% of hours above $100\n"
        f"{pct_negative:.1f}% of hours negative",
        transform=axes[1, 0].transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    axes[1, 0].set_title('Price Distribution (clipped at $150 for readability)')
    axes[1, 0].set_xlabel('Price ($/MWh)')
    axes[1, 0].set_ylabel('Number of Hours')
    axes[1, 0].set_xlim(-10, 150)
    axes[1, 0].set_xticks([0, 10, 20, 30, 40, 50, 75, 100, 125, 150])
    axes[1, 0].legend(loc='upper right', bbox_to_anchor=(1, 0.72))

    # ── Plot 4: Price regime over time ────────────────────────────────────────
    # Each hour plotted as a point colored by market regime:
    # Routine  : price <= $50   (85.4% of hours)
    # Elevated : $50-$200       (12.8% of hours)
    # Spike    : price > $200   (1.6%  of hours)
    df_sorted = df.copy()
    df_sorted['datetime'] = pd.to_datetime(df_sorted['datetime'])
    df_sorted = df_sorted.sort_values('datetime')
    df_sorted['price_plot'] = df_sorted['price_mwh'].clip(upper=500)

    routine_df  = df_sorted[df_sorted['price_mwh'] <= 50]
    elevated_df = df_sorted[(df_sorted['price_mwh'] > 50) & (df_sorted['price_mwh'] <= 200)]
    spike_df    = df_sorted[df_sorted['price_mwh'] > 200]

    axes[1, 1].scatter(routine_df['datetime'],  routine_df['price_plot'],
                       color='#2A7F8F', alpha=0.3, s=0.5, label='Routine (<=$50)')
    axes[1, 1].scatter(elevated_df['datetime'], elevated_df['price_plot'],
                       color='#f0a500', alpha=0.6, s=0.8, label='Elevated ($50-$200)')
    axes[1, 1].scatter(spike_df['datetime'],    spike_df['price_plot'],
                       color='#d94f3d', alpha=0.9, s=1.5, label='Spike (>$200)')

    axes[1, 1].set_title('Market Regimes Over Time (y-axis capped at $500)')
    axes[1, 1].set_ylabel('Price ($/MWh)')
    axes[1, 1].set_xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2026-07-01'))
    axes[1, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=30)
    legend = axes[1, 1].legend(loc='upper right', markerscale=6)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ercot_eda.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved ercot_eda.png")


def print_data_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA SUMMARY: ERCOT HB_NORTH DAM Prices (2019-2026)")
    print("=" * 60)

    # ── Overview ──────────────────────────────────────────────────
    print("\nOVERVIEW")
    print(f"  Total hours       : {len(df):,}")
    print(f"  Date range        : {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    print(f"  Years covered     : {df['year'].nunique()} ({df['year'].min()} - {df['year'].max()})")

    # ── Price stats ───────────────────────────────────────────────
    print("\nPRICE STATISTICS ($/MWh)")
    print(f"  Mean              : ${df['price_mwh'].mean():.2f}")
    print(f"  Median            : ${df['price_mwh'].median():.2f}")
    print(f"  Std deviation     : ${df['price_mwh'].std():.2f}")
    print(f"  Min               : ${df['price_mwh'].min():.2f}")
    print(f"  Max               : ${df['price_mwh'].max():,.2f}")
    print(f"  25th percentile   : ${df['price_mwh'].quantile(0.25):.2f}")
    print(f"  75th percentile   : ${df['price_mwh'].quantile(0.75):.2f}")
    print(f"  99th percentile   : ${df['price_mwh'].quantile(0.99):.2f}")
    print(f"  99.9th percentile : ${df['price_mwh'].quantile(0.999):.2f}")

    # ── Extreme events ────────────────────────────────────────────
    print("\nEXTREME EVENTS")
    neg   = (df['price_mwh'] < 0).sum()
    s100  = (df['price_mwh'] > 100).sum()
    s200  = (df['price_mwh'] > 200).sum()
    s500  = (df['price_mwh'] > 500).sum()
    uri   = df[(df['datetime'].dt.year == 2021) & (df['datetime'].dt.month == 2)]['price_mwh'].max()
    print(f"  Negative hours    : {neg} ({neg/len(df)*100:.2f}% of all hours)")
    print(f"  Hours above $100  : {s100} ({s100/len(df)*100:.1f}% of all hours)")
    print(f"  Hours above $200  : {s200} ({s200/len(df)*100:.1f}% of all hours)")
    print(f"  Hours above $500  : {s500} ({s500/len(df)*100:.1f}% of all hours)")
    print(f"  Feb 2021 URI peak : ${uri:,.2f}/MWh")

    # ── Hourly patterns ───────────────────────────────────────────
    print("\nMEDIAN PRICE BY HOUR ($/MWh)")
    hourly = df.groupby('hour')['price_mwh'].median()
    print(f"  Cheapest hour     : Hour {hourly.idxmin():02d}:00  ${hourly.min():.2f}")
    print(f"  Most expensive    : Hour {hourly.idxmax():02d}:00  ${hourly.max():.2f}")
    print(f"  Peak/off-peak gap : ${hourly.max() - hourly.min():.2f}/MWh")

    # ── Monthly patterns ──────────────────────────────────────────
    print("\nMEDIAN PRICE BY MONTH ($/MWh)")
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly = df.groupby('month')['price_mwh'].median()
    for m, price in monthly.items():
        flag = " <-- summer peak" if m in [6,7,8,9] else ""
        print(f"  {month_names[m]}  : ${price:.2f}{flag}")

    # ── Weekend vs weekday ────────────────────────────────────────
    print("\nWEEKDAY vs WEEKEND")
    weekday_med = df[df['is_weekend'] == 0]['price_mwh'].median()
    weekend_med = df[df['is_weekend'] == 1]['price_mwh'].median()
    print(f"  Weekday median    : ${weekday_med:.2f}/MWh")
    print(f"  Weekend median    : ${weekend_med:.2f}/MWh")
    print(f"  Weekend discount  : ${weekday_med - weekend_med:.2f}/MWh")

    # ── Distribution buckets ──────────────────────────────────────
    print("\nPRICE DISTRIBUTION")
    buckets = [
        ("Below $0 (negative)", df['price_mwh'] < 0),
        ("$0 - $25",   (df['price_mwh'] >= 0)   & (df['price_mwh'] < 25)),
        ("$25 - $50",  (df['price_mwh'] >= 25)  & (df['price_mwh'] < 50)),
        ("$50 - $100", (df['price_mwh'] >= 50)  & (df['price_mwh'] < 100)),
        ("$100 - $200",(df['price_mwh'] >= 100) & (df['price_mwh'] < 200)),
        ("Above $200", df['price_mwh'] >= 200),
    ]
    for label, mask in buckets:
        count = mask.sum()
        pct   = count / len(df) * 100
        print(f"  {label:<22}: {count:>6,} hours ({pct:.1f}%)")

    print("\n" + "=" * 60)


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
        print("No differencing needed before modeling.")
    else:
        print("Result: NON-STATIONARY (p > 0.05)")
        print("Consider first-differencing before applying SARIMA.")


if __name__ == '__main__':
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])
    print(f"Rows: {len(df):,} | {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    print_data_summary(df)

    print("\nGenerating EDA plots...")
    plot_eda(df)

    run_stationarity_test(df)
