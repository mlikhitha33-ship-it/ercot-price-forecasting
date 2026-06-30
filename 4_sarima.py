"""
4_sarima.py
-----------
Seasonal Naive baseline model for ERCOT day-ahead price forecasting.

Reads  : ercot_real_prices.csv   (output of 1_load_data.py)
Outputs: baseline_forecast.png
Prints : MAE, RMSE, MAPE on hourly 2025-2026 test data

Evaluated on the same test period and hourly granularity as the LSTM
in 5_lstm.py, so the two sets of metrics are directly comparable.

SARIMA was attempted first but proved unsuitable for ERCOT prices.
It consistently overestimated prices regardless of how the February
2021 URI spike was handled in training. SARIMA fits parameters to
the entire series and ERCOT prices are too volatile and non-linear
for that to forecast reliably on this dataset.

The Seasonal Naive baseline used here predicts each hour's price as
the same hour one week (168 hours) earlier. No training required and
it works because ERCOT weekly demand patterns are highly consistent.

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


def run_naive_baseline(df: pd.DataFrame) -> dict:
    # Use hourly prices directly, not daily median.
    # Evaluated on the same test period and granularity as the LSTM
    # (hourly 2025-2026 data, 24-hour-ahead forecasts) so the two
    # models are directly comparable.
    hourly = df.set_index('datetime')['price_mwh'].sort_index()

    test_mask = hourly.index >= '2025-01-01'
    test = hourly[test_mask]

    # Seasonal naive: predict each hour's price as the same hour
    # exactly one week (168 hours) earlier. This mirrors the LSTM's
    # price_lag_168h feature, which is the single strongest signal
    # in this dataset.
    forecast = hourly.shift(168)
    forecast_test = forecast[test_mask].dropna()
    test_aligned  = test.loc[forecast_test.index]

    mae  = mean_absolute_error(test_aligned, forecast_test)
    rmse = np.sqrt(mean_squared_error(test_aligned, forecast_test))
    mape = np.mean(np.abs((test_aligned - forecast_test) / (test_aligned.abs() + 1))) * 100

    print(f"Test : {test_aligned.index[0]} to {test_aligned.index[-1]} ({len(test_aligned):,} hours)")
    print("\nBaseline: Seasonal Naive (predict = same hour, 168h / 1 week earlier)")
    print("Evaluated on hourly 2025-2026 data, same test period as the LSTM")

    return {
        'test':     test_aligned,
        'forecast': forecast_test,
        'mae':      mae,
        'rmse':     rmse,
        'mape':     mape,
    }


def plot_baseline(results: dict) -> None:
    test     = results['test']
    forecast = results['forecast']
    mae      = results['mae']
    rmse     = results['rmse']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Seasonal Naive Baseline: ERCOT HB_NORTH Hourly Price Forecast (2025-2026)',
                 fontsize=13, fontweight='bold')

    # Full test period
    axes[0].plot(test.index, test.values,
                 label='Actual', color='#333333', linewidth=0.6)
    axes[0].plot(forecast.index, forecast.values,
                 label='Naive Forecast (168h / 1 week earlier)', color='#d94f3d',
                 linewidth=0.6, linestyle='--', alpha=0.8)
    axes[0].set_title(f'Full Test Period 2025-2026  |  MAE=${mae:.2f}  |  RMSE=${rmse:.2f}')
    axes[0].set_ylabel('Price ($/MWh)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Zoom in on first 7 days for hourly detail
    n_hours = 24 * 7
    axes[1].plot(test.index[:n_hours], test.values[:n_hours],
                 label='Actual', color='#333333', linewidth=1.5, marker='o', markersize=2)
    axes[1].plot(forecast.index[:n_hours], forecast.values[:n_hours],
                 label='Naive Forecast', color='#d94f3d', linewidth=1.5,
                 linestyle='--', marker='s', markersize=2)
    axes[1].set_title('First 7 Days of Test Period, Hourly Detail')
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

    results = run_naive_baseline(df)

    print("\nBaseline Results (Test: 2025-2026, hourly)")
    print("-" * 45)
    print("Method: Seasonal Naive (same hour, 1 week earlier)")
    print(f"MAE  : ${results['mae']:.2f}/MWh")
    print(f"RMSE : ${results['rmse']:.2f}/MWh")
    print(f"MAPE : {results['mape']:.1f}%")
    print("-" * 45)
    print("Same test period and hourly granularity as the LSTM (5_lstm.py)")
    print("so these numbers are directly comparable.")

    plot_baseline(results)
