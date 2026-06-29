# Can a Machine Learn the Price of Electricity?

*A data science project using real ERCOT market data, classical statistics and deep learning to forecast Texas wholesale electricity prices.*

---

## The Problem

### What is ERCOT?

ERCOT (Electric Reliability Council of Texas) is the independent system operator that manages electric power across most of Texas. It operates one of the few fully deregulated electricity markets in the US, serving roughly 26 million customers and about 90% of the state's electric load.

Unlike regulated markets where utilities set prices, ERCOT runs a competitive wholesale market where prices are determined each hour by supply and demand. Generators, retailers, traders and storage operators all participate, buying and selling power based on market conditions.

**How is price data collected?**
ERCOT calculates Day-Ahead Market (DAM) Settlement Point Prices for every hour of the following day. Generators submit offers, load serving entities submit bids, and ERCOT's algorithms find the price that balances supply and demand at each grid location. Results are published publicly on ERCOT's website each day.

**Who uses this data?**
Primary users are energy traders, grid-scale battery storage operators, renewable energy developers, retail electricity providers and analytics firms building forecasting tools for the energy industry.

### The Business Problem

Battery storage operators in ERCOT use a strategy called **price arbitrage**:

- **Charge** when prices are low, typically overnight between midnight and 6am ($15-25/MWh)
- **Discharge and sell** when prices are high, typically during the evening peak between 5-8pm ($50-80/MWh)
- Profit is the spread between those two prices minus operating costs

Forecast accuracy directly determines dispatch quality. Missing the evening ramp by two hours costs real revenue. This project asks: can we build a reliable 24-hour ahead price forecast for the ERCOT North Hub?

---

## The Data

I pulled **ERCOT Day-Ahead Market Settlement Point Prices** for the `HB_NORTH` hub, the North Texas benchmark price used by traders and storage operators across the state.

**Source:** [ERCOT Historical DAM Load Zone and Hub Prices](https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er)

The dataset runs from **January 2019 through June 2026**, 65,464 hourly observations. Three things make this dataset hard to model:

- **460 hours** where prices exceeded $500/MWh
- **69 hours** of negative prices from wind oversupply and curtailment
- **February 2021**: the URI winter storm pushed prices to **$8,998.99/MWh**, nearly 200 times a typical price

These are not anomalies to remove. They are the hours that matter most to storage operators and any useful model has to deal with them.

---

## What the Data Showed Before Any Modeling

I spent time looking at the data before writing a single line of model code. Three patterns came up consistently.

**Hour of day matters a lot.** Prices are lowest between midnight and 5am when industrial demand falls and wind generation is typically high. They climb during morning ramp (7-9am) and again during the evening peak (5-8pm). Hour-of-day had to be a feature.

**Texas summers push prices up persistently.** June through September show higher median prices driven by air conditioning load. A model trained only on winter data would not generalize well to summer months.

**The price distribution has a long right tail.** Most hours sit between $20-$60/MWh but spike events pull the distribution hard to the right. Standard loss functions like MSE get pulled toward predicting these extremes, which creates problems during training.

---

## Two Models, One Dataset

I started with a classical statistical model before moving to deep learning. If a decades-old method outperforms a neural network, the issue is not model choice but data, features or training setup. SARIMA gives an honest baseline to measure against.

---

### Model 1: SARIMA

SARIMA (Seasonal AutoRegressive Integrated Moving Average) captures three things in a time series: how past values predict the future (autoregression), how past forecast errors carry forward (moving average) and repeating seasonal cycles. It has been standard in energy forecasting for decades.

Before fitting, I ran an **Augmented Dickey-Fuller test** to check stationarity. SARIMA assumes the statistical properties of the series don't drift over time. The ADF result (statistic = -9.64, p-value = 0.000) confirmed the series is stationary, so no differencing was needed.

Trained on daily median prices from 2019-2022, tested on 2023.

---

### Model 2: LSTM (PyTorch)

An LSTM (Long Short-Term Memory network) is a recurrent neural network built for sequential data. Unlike SARIMA which assumes a fixed linear structure, an LSTM learns non-linear relationships across long time windows. Tuesday evening prices this week often look like Tuesday evening prices last week, and LSTMs can learn that kind of pattern.

**Architecture:**
```
Input  : 48-hour sliding window x 20 features
LSTM   : 2 layers, hidden size 128, dropout 0.2
FC head: 128 -> ReLU(64) -> 24
Output : next 24-hour price forecast
Parameters: 218,712
```

**Features (20 total):**
- Lagged prices: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
- Rolling stats: 24h mean/std, 7-day mean/std
- Cyclical time encoding: sin/cos for hour, month, day-of-week
- Calendar: is_weekend

On cyclical encoding: hour of day is encoded with sine and cosine rather than raw integers. Hour 23 and hour 0 are adjacent in real life but 23 apart numerically. Sin/cos wraps the cycle so the model sees them as neighbors.

**Training choices:**
- **Huber loss** instead of MSE. With a $8,999/MWh spike in the training data, MSE would pull the model heavily toward predicting extremes.
- **Winsorized prices** at the 99.9th percentile before scaling. Without this, MinMaxScaler compresses normal-range prices into a narrow band near zero.
- **Chronological train/val/test split**: 2019-2023 train, 2024 validation, 2025-2026 test. Time series data cannot be randomly shuffled without leaking future information into training.
- **Gradient clipping** at max_norm=1.0 to stabilize training on a volatile series.

---

## Results

| Metric | SARIMA | LSTM |
|---|---|---|
| MAE | $22.06/MWh | $25.53/MWh |
| RMSE | $23.22/MWh | $70.13/MWh |
| MAPE | 94% | 139% |

### What each metric measures

**MAE (Mean Absolute Error)** is the average absolute difference between predicted and actual prices across all test hours. If the model predicts $40/MWh and actual is $55/MWh, that hour contributes $15. Every hour is weighted equally. MAE tells you: on a typical hour, how far off is the forecast?

**RMSE (Root Mean Squared Error)** squares each error before averaging, then takes the square root. Large errors are penalized much more than small ones. A $500 error contributes 25 times more to RMSE than a $100 error. RMSE tells you: when the model is wrong by a lot, how wrong does it get?

**MAPE (Mean Absolute Percentage Error)** expresses errors as a percentage of the actual price. A $10 error on a $20/MWh hour is 50%; the same $10 error on a $100/MWh hour is 10%. MAPE tells you: how large are errors relative to the price level?

### Reading the comparison

**MAE: SARIMA won ($22.06 vs $25.53).** On a typical hour, SARIMA's forecast was $3.47 closer to actual. For routine non-extreme hours, the classical model captures weekly and daily patterns well enough to beat the LSTM at this training scale.

**RMSE: the gap is large ($23.22 vs $70.13).** The LSTM's RMSE is three times higher. Since RMSE penalizes large errors heavily, this points to the LSTM making significant mistakes on spike hours while SARIMA's errors stay more contained.

**MAPE is high for both models.** This is expected with ERCOT data. When prices drop to $5/MWh during curtailment or spike to $500/MWh during scarcity events, even a reasonable absolute error produces a large percentage error. High MAPE is a known characteristic of volatile electricity markets.

**What this means in practice:** The LSTM's MAE is close to SARIMA but its RMSE shows a real weakness on price spikes. Spike hours represent the highest-value dispatch opportunities for storage operators, so a model that performs well on average but fails on extremes has limited value for the actual use case.

---

## What I'd Do Differently

**Two-stage modeling for spikes.** One model trying to predict both $25/MWh routine hours and $8,999/MWh crisis hours is doing two different jobs. A spike classifier followed by a separate price regressor for each regime would likely close most of the RMSE gap.

**Add external market features.** This model only sees historical prices and time. Prices are driven by supply-demand fundamentals: wind generation, ERCOT load forecasts, natural gas futures. These features would likely improve accuracy more than any architecture change.

**Prediction intervals over point forecasts.** A storage operator needs to know not just what the price will be but how confident the model is. Quantile regression or Monte Carlo Dropout would produce intervals that enable better risk-adjusted dispatch decisions.

**Walk-forward cross-validation.** A single train/test split reflects performance in one time window. Rolling walk-forward validation across multiple periods gives a more reliable picture of how the model generalizes.

---

## How to Run

### Google Colab (recommended)

Colab provides a free GPU that reduces LSTM training from a few hours on local CPU to around 10 minutes.

**Step 1: Download the ERCOT data**

Go to [ERCOT Historical DAM Load Zone and Hub Prices](https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er) and download the annual xlsx files. Each file covers one full year with one tab per month. The files download as zip archives — extract them before uploading.

**Step 2: Set up your Google Drive folder**

Create this exact structure in Google Drive:

```
My Drive/
└── ercot-price-forecasting/
    ├── ercot_raw/
    │   ├── rpt...DAMLZHBSPP_2019.xlsx
    │   ├── rpt...DAMLZHBSPP_2020.xlsx
    │   ├── rpt...DAMLZHBSPP_2021.xlsx
    │   ├── rpt...DAMLZHBSPP_2022.xlsx
    │   ├── rpt...DAMLZHBSPP_2023.xlsx
    │   ├── rpt...DAMLZHBSPP_2024.xlsx
    │   ├── rpt...DAMLZHBSPP_2025.xlsx
    │   └── rpt...DAMLZHBSPP_2026.xlsx
    ├── 1_load_data.py
    ├── 2_eda.py
    ├── 3_features.py
    ├── 4_sarima.py
    ├── 5_lstm.py
    └── requirements.txt
```

**Step 3: Open a new notebook at [colab.research.google.com](https://colab.research.google.com) and run these cells in order:**

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Navigate to project folder
import os
os.chdir('/content/drive/MyDrive/ercot-price-forecasting')
os.listdir()  # confirm your files are visible
```

```python
# Cell 3: Install dependencies
!pip install -r requirements.txt -q
```

```python
# Cell 4: Run scripts in order
!python 1_load_data.py
!python 2_eda.py
!python 3_features.py
!python 4_sarima.py
!python 5_lstm.py
```

**Step 4: View charts inline**

```python
from IPython.display import Image
Image('ercot_eda.png')          # after 2_eda.py
Image('sarima_forecast.png')    # after 4_sarima.py
Image('lstm_24h_forecasts.png') # after 5_lstm.py
```

Each script saves its output as a CSV or PNG back to your Drive folder so results persist between sessions.

---



[GitHub](https://github.com/mlikhitha33-ship-it)
