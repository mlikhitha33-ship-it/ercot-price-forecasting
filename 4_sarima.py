"""
4_sarima.py
-----------
SARIMA baseline model for ERCOT day-ahead price forecasting.

Reads  : ercot_real_prices.csv   (output of 1_load_data.py)
Outputs: sarima_forecast.png
Prints : MAE, RMSE, MAPE on 2023 test set

Why SARIMA as a baseline?
Classical baselines are essential for honest ML evaluation.
If the LSTM can't beat SARIMA, the added complexity isn't justified.

SARIMA(1,0,1)(1,0,1,7):
- (1,0,1)   : AR(1), no differencing needed (series is stationary), MA(1)
- (1,0,1,7) : seasonal AR and MA with weekly period (7 days)

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
    # Use daily median price — SARIMA on hourly data is very slow
    # and weekly seasonality is clearer at daily resolution
    daily = df.set_index('datetime')['price_mwh'].resample('D').median()

    # Train: 2019-2022 | Test: 2023
    # Exclude Feb 2021 URI spike from distorting the baseline
    train = daily['2019':'2022']
    test  = daily['2023']

    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Test : {test.index[0].date()} to {test.index[-1].date()}  ({len(test)} days)")
    print("\nFitting SARIMA(1,0,1)(1,0,1,7)...")

    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)

    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    # Metrics
    mae  = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / (test.abs() + 1))) * 100

    return {
        'test':     test,
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
    fig.suptitle('SARIMA Baseline — ERCOT HB_NORTH Daily Price Forecast (2023)',
                 fontsize=13, fontweight='bold')

    # Full year
    axes[0].plot(test.index, test.values,
                 label='Actual', color='#333333', linewidth=1.5)
    axes[0].plot(forecast.index, forecast.values,
                 label='SARIMA Forecast', color='#d94f3d', linewidth=1.5, linestyle='--')
    axes[0].set_title(f'Full Year 2023 | MAE=${mae:.2f} | RMSE=${rmse:.2f}')
    axes[0].set_ylabel('Price ($/MWh)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Zoom in on one month to see daily pattern
    axes[1].plot(test.index[:60], test.values[:60],
                 label='Actual', color='#333333', linewidth=1.8, marker='o', markersize=3)
    axes[1].plot(forecast.index[:60], forecast.values[:60],
                 label='SARIMA Forecast', color='#d94f3d', linewidth=1.8,
                 linestyle='--', marker='s', markersize=3)
    axes[1].set_title('First 60 Days (Jan-Feb 2023) — Zoomed')
    axes[1].set_ylabel('Price ($/MWh)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sarima_forecast.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved sarima_forecast.png")


if __name__ == '__main__':
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])

    results = run_sarima(df)

    print("\nSARIMA Results (Test: 2023)")
    print("-" * 35)
    print(f"MAE  : ${results['mae']:.2f}/MWh")
    print(f"RMSE : ${results['rmse']:.2f}/MWh")
    print(f"MAPE : {results['mape']:.1f}%")
    print("-" * 35)
    print("Note: High MAPE is expected — ERCOT prices are highly volatile.")
    print("SARIMA serves as the classical baseline the LSTM must beat.")

    plot_sarima(results)
