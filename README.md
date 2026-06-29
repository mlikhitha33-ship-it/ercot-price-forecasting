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

Getting the timing right matters. Missing the evening ramp by two hours is the difference between a good day and a bad one. That is the forecasting problem this project works on.

---

## Exploratory Data Analysis

I pulled **ERCOT Day-Ahead Market Settlement Point Prices** for the `HB_NORTH` hub, the North Texas benchmark price used by traders and storage operators across the state.

**Source:** [ERCOT Historical DAM Load Zone and Hub Prices](https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er)

The dataset runs from **January 2019 through June 2026**, 65,464 hourly observations. A few characteristics make this market particularly difficult to model:

- **460 hours** where prices exceeded $500/MWh
- **69 hours** of negative prices. This happens when wind generation is high and demand is low, typically overnight. Wind generators sometimes pay to offload power rather than shut down because federal production tax credits make it profitable even at a negative price. ERCOT's market allows this and it shows up in the data like any other hour.
- **February 2021**: the URI winter storm pushed prices to **$8,998.99/MWh**, nearly 200 times a typical price

Removing these hours would make the modeling problem cleaner but less useful. Storage operators care most about exactly these events.

---

## Key Findings

Running `2_eda.py` produces the following data summary:

```
OVERVIEW
  Total hours       : 65,464
  Date range        : 2019-01-01 to 2026-06-20

PRICE STATISTICS ($/MWh)
  Mean              : $54.00
  Median            : $23.75
  Std deviation     : $347.20
  Min               : $-6.00
  Max               : $8,998.99
  25th percentile   : $17.26
  75th percentile   : $36.76
  99th percentile   : $331.61

EXTREME EVENTS
  Negative hours    : 69 (0.11% of all hours)
  Hours above $100  : 2,588 (4.0% of all hours)
  Hours above $500  : 460 (0.7% of all hours)
  Feb 2021 URI peak : $8,998.99/MWh

MEDIAN PRICE BY HOUR ($/MWh)
  Cheapest hour     : Hour 03:00  $17.38
  Most expensive    : Hour 19:00  $38.87
  Peak/off-peak gap : $21.49/MWh

MEDIAN PRICE BY MONTH ($/MWh)
  Jan: $22.23  Feb: $19.77  Mar: $21.13  Apr: $21.64
  May: $23.05  Jun: $25.20  Jul: $26.82  Aug: $29.43
  Sep: $26.27  Oct: $25.95  Nov: $25.44  Dec: $22.60

WEEKDAY vs WEEKEND
  Weekday median    : $24.16/MWh
  Weekend median    : $22.66/MWh
  Weekend discount  : $1.50/MWh

PRICE DISTRIBUTION
  Below $0 (negative)   :     69 hours (0.1%)
  $0 - $25              : 35,105 hours (53.6%)
  $25 - $50             : 20,836 hours (31.8%)
  $50 - $100            :  6,837 hours (10.4%)
  $100 - $200           :  1,546 hours (2.4%)
  Above $200            :  1,071 hours (1.6%)
```

A few things stood out from these numbers.

**Hour of day matters more than expected.** Prices sit around $17/MWh between 2-4am and climb to $37-39/MWh between 5-7pm. That is more than a 2x swing within a single day, which makes hour-of-day one of the most important features in the model.

**August is consistently the most expensive month.** June through September are all elevated from air conditioning load, but August median prices hit nearly $30/MWh compared to $17-22/MWh in winter. A model without seasonal features would struggle badly on summer data.

**The market runs in two modes.** 85.4% of hours sit below $50/MWh — the routine market, predictable and weekly-patterned. The other 14.6% are elevated or spike hours, more concentrated in 2022 and 2023. The regime chart below makes this visible. Any model has to deal with both modes and they behave very differently.

![ERCOT EDA](ercot_eda.png)

---

## Data Cleaning and Preparation

**Filtering to one settlement point**

The raw ERCOT files contain 15 rows per hour — one price per settlement point. Loading all of them would mix 15 different price series together. HB_NORTH (North Hub) covers the Dallas/Fort Worth region and is the most widely referenced benchmark in ERCOT trading. Filtering to HB_NORTH brought the dataset from nearly a million rows down to 65,464 — one row per hour.

**Duplicate hours**

ERCOT marks DST fall-back hours with a "Repeated Hour Flag." Eight duplicate rows were dropped across the full dataset, one per year where clocks fall back.

**Missing hours**

Eight hours are missing across 7.5 years — the DST spring-forward hours where clocks skip from 2am to 3am. The hour does not exist in the market so no imputation was done.

**Negative prices**

69 hours have negative prices, the lowest at -$6.00/MWh. These were kept. They are real market events and removing them would teach the model that prices never go negative, which is not true.

**The URI spike**

The February 2021 URI storm produced 460+ hours above $500/MWh. These were kept but handled carefully. For the LSTM, prices were winsorized at the 99.9th percentile ($7,556/MWh) before scaling. This is not removing the spike — it stops MinMaxScaler from compressing all normal-range prices into a band so narrow the model cannot distinguish between a $20 hour and a $60 hour.

**Timestamp conversion**

ERCOT publishes prices using an hour-ending convention — "Hour Ending 01:00" means the hour from midnight to 1am. All timestamps were converted to hour-starting by subtracting one hour, which is the standard convention for time series modeling.

---

## Modeling

SARIMA was the first model tried. It is a classical statistical model that works well on smooth, seasonal time series — monthly retail sales, airline passenger counts and similar. ERCOT prices are a different problem. The weekly rhythm is consistent but the same dataset also has hours where prices jump from $25 to $500 within a single day. SARIMA fits one set of linear parameters to the entire series and when spikes are present those parameters get pulled toward the extremes. The result was a 2023 forecast that consistently ran $30-40/MWh above actual prices regardless of how the URI spike was handled in training. SARIMA was dropped.

---

### Model 1: Seasonal Naive Baseline

Instead of SARIMA, a simpler baseline was used: predict each day's price as the same day of the prior week. No training required and it works because ERCOT weekly demand patterns are consistent — Tuesday evening this week looks a lot like Tuesday evening last week.

The chart below shows how this forecast tracked actual 2023 prices.

![Baseline Forecast](baseline_forecast.png)

It follows the weekly rhythm well and stays in the right price range. Where it misses — sudden spikes — is expected. It has no awareness of current market conditions, only what happened last week.

---

### Model 2: LSTM (PyTorch)

An LSTM (Long Short-Term Memory network) is a recurrent neural network built for sequential data. It learns non-linear relationships across long time windows — something SARIMA cannot do. Tuesday evening prices this week often look a lot like Tuesday evening prices last week, and LSTMs are designed to pick up on exactly that kind of pattern.

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

On cyclical encoding: hour of day is encoded using sine and cosine rather than raw integers. Hour 23 and hour 0 are adjacent in real life but 23 apart numerically. Sin/cos wraps the cycle so the model treats them as neighbors.

**Training choices:**
- **Huber loss** instead of MSE. With a $8,999/MWh spike in the training data, MSE would pull the model heavily toward predicting extremes.
- **Winsorized prices** at the 99.9th percentile before scaling. Without this, MinMaxScaler compresses normal-range prices into a very narrow band near zero.
- **Chronological split**: 2019-2023 train, 2024 validation, 2025-2026 test. Time series data cannot be randomly shuffled without leaking future information into training.
- **Gradient clipping** at max_norm=1.0 to stabilize training on a volatile series.

---

## Results

| Metric | Naive Baseline | LSTM |
|---|---|---|
| MAE | $5.81/MWh | $25.53/MWh |
| RMSE | $8.92/MWh | $70.13/MWh |
| MAPE | 26.5% | 139% |

### What each metric measures

**MAE (Mean Absolute Error)** is the average absolute difference between predicted and actual prices across all test hours. If the model predicts $40/MWh and actual is $55/MWh, that hour contributes $15. Every hour is weighted equally. On a typical hour, how far off is the forecast?

**RMSE (Root Mean Squared Error)** squares each error before averaging then takes the square root. Large errors are penalized much more than small ones. A $500 error contributes 25 times more to RMSE than a $100 error. When the model is wrong by a lot, how wrong does it get?

**MAPE (Mean Absolute Percentage Error)** expresses errors as a percentage of actual price. A $10 error on a $20/MWh hour is 50%; the same $10 error on a $100/MWh hour is 10%. How large are the errors relative to the price level?

### Reading the numbers

The naive baseline won on every metric. MAE of $5.81 vs the LSTM's $25.53, RMSE of $8.92 vs $70.13.

The naive baseline works well here because ERCOT weekly patterns are strong — the same day last week is a genuinely good predictor for most hours. The LSTM trained on 10 epochs with a 48-hour lookback could not beat that. The RMSE being nearly 8 times higher points to the spike problem — the LSTM makes large errors on high-price hours while the naive forecast, anchored to last week, misses them in a more contained way.

The LSTM needs more work before it adds value over a simple baseline. The next section covers what that looks like.

---

## What I'd Do Differently

**Build a two-stage model for spikes.** One model trying to handle both $25/MWh routine hours and $8,999/MWh crisis hours is doing two different jobs. A spike classifier followed by a separate price regressor for each regime would likely close most of the RMSE gap.

**Add external market features.** This model only sees historical prices and time. Prices are driven by supply-demand fundamentals: wind generation, ERCOT load forecasts, natural gas futures. Adding those would likely improve accuracy more than any architecture change.

**Prediction intervals instead of point forecasts.** A storage operator needs to know not just what the price will be but how confident the model is. Quantile regression or Monte Carlo Dropout would give prediction intervals that support better risk-adjusted dispatch decisions.

**Walk-forward cross-validation.** A single train/test split reflects performance in one time window. Rolling walk-forward validation across multiple periods gives a more reliable picture of how the model holds up across different market conditions.

---

## How to Run

### Google Colab (recommended)

Colab provides a free GPU that reduces LSTM training from a few hours on local CPU to around 10 minutes.

**Step 1: Download the ERCOT data**

Go to [ERCOT Historical DAM Load Zone and Hub Prices](https://www.ercot.com/mp/data-products/data-product-details?id=np4-180-er) and download the annual xlsx files. Each file covers one full year with one tab per month. The files download as zip archives so extract them before uploading.

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
os.listdir()
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
Image('baseline_forecast.png')  # after 4_sarima.py
Image('lstm_24h_forecasts.png') # after 5_lstm.py
```

Each script saves output back to your Drive folder so results persist between sessions.

---

