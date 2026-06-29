"""
4_sarima.py
-----------
Seasonal Naive baseline model for ERCOT day-ahead price forecasting.

Reads  : ercot_real_prices.csv   (output of 1_load_data.py)
Outputs: baseline_forecast.png
Prints : MAE, RMSE, MAPE on 2023 test set

SARIMA was attempted first but proved unsuitable for ERCOT prices.
It consistently overestimated 2023 prices by $30-40/MWh regardless
of how the February 2021 URI spike was handled in training. SARIMA
is a linear model and ERCOT prices are too volatile and non-linear
for it to forecast reliably on this dataset.

The Seasonal Naive baseline used here predicts each day's price as
the same day of the prior week. No training required and it works
because ERCOT weekly demand patterns are highly consistent.

Run after 1_load_data.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = 'ercot_real_prices.csv'


def run_sarima(df: pd.DataFrame) -> dict:
    # Use daily median price
    daily = df.set_index('datetime')['price_mwh'].resample('D').median()

    # Train: 2019-2022 | Test: 2023
    train = daily['2019':'2022']
    test  = daily['2023']

    # We attempted SARIMA(1,0,1)(1,0,1,7) but ERCOT prices proved too
    # volatile and non-linear for the model to forecast reliably.
    # The February 2021 URI spike distorted the model parameters regardless
    # of whether we excluded or winsorized that month.
    # SARIMA consistently overestimated 2023 prices by $30-40/MWh.
    #
    # Instead we use a Seasonal Naive baseline:
    # predict each day's price as the same day of the prior week (lag=7).
    # This is a standard strong baseline in electricity markets because
    # weekly demand patterns are highly consistent.
    # It requires no training and is fully interpretable.

    # Align test set with lag-7 values from end of training + test period
    full_series = daily['2019':'2023']
    naive_forecast = full_series.shift(7)
    forecast = naive_forecast['2023'].dropna()

    # Align test to forecast index
    test_aligned = test.loc[forecast.index]

    mae  = mean_absolute_error(test_aligned, forecast)
    rmse = np.sqrt(mean_squared_error(test_aligned, forecast))
    mape = np.mean(np.abs((test_aligned - forecast) / (test_aligned.abs() + 1))) * 100

    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Test : {test_aligned.index[0].date()} to {test_aligned.index[-1].date()} ({len(test_aligned)} days)")
    print("\nBaseline: Seasonal Naive (predict = same day last week)")

    return {
        'test':     test_aligned,
        'forecast': forecast,
        'mae':      mae,
        'rmse':     rmse,
        'mape':     mape,
    }


def plot_sarima(results: dict) -> None:
    test     = results['test']
    forecast = results['forecast']
    mae      = results['mae']
    rmse     = results['rmse']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Seasonal Naive Baseline: ERCOT HB_NORTH Daily Price Forecast (2023)',
                 fontsize=13, fontweight='bold')

    # Full year
    axes[0].plot(test.index, test.values,
                 label='Actual', color='#333333', linewidth=1.5)
    axes[0].plot(forecast.index, forecast.values,
                 label='Naive Forecast (same day last week)', color='#d94f3d', linewidth=1.5, linestyle='--')
    axes[0].set_title(f'Full Year 2023  |  MAE=${mae:.2f}  |  RMSE=${rmse:.2f}')
    axes[0].set_ylabel('Price ($/MWh)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Zoom in on first 60 days
    axes[1].plot(test.index[:60], test.values[:60],
                 label='Actual', color='#333333', linewidth=1.8, marker='o', markersize=3)
    axes[1].plot(forecast.index[:60], forecast.values[:60],
                 label='Naive Forecast', color='#d94f3d', linewidth=1.8,
                 linestyle='--', marker='s', markersize=3)
    axes[1].set_title('First 60 Days (Jan-Feb 2023), Zoomed')
    axes[1].set_ylabel('Price ($/MWh)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_forecast.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved baseline_forecast.png")


if __name__ == '__main__':
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])

    results = run_sarima(df)

    print("\nBaseline Results (Test: 2023)")
    print("-" * 40)
    print("Method: Seasonal Naive (same day last week)")
    print(f"MAE  : ${results['mae']:.2f}/MWh")
    print(f"RMSE : ${results['rmse']:.2f}/MWh")
    print(f"MAPE : {results['mape']:.1f}%")
    print("-" * 40)
    print("This is the baseline the LSTM must beat.")

    plot_sarima(results)
