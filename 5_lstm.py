"""
5_lstm.py
---------
2-layer LSTM (PyTorch) for 24-hour ahead ERCOT price forecasting.

Reads  : ercot_features.csv     (output of 3_features.py)
Outputs: best_lstm_model.pt     (saved model weights)
         lstm_training_curve.png
         lstm_24h_forecasts.png
         lstm_horizon_mae.png
Prints : MAE, RMSE, MAPE on test set (2025-2026)

Architecture:
- Input  : 48-hour sliding window of 20 features
- LSTM   : 2 layers, hidden size 128, dropout 0.2
- Output : next 24-hour price forecast

Key decisions:
- Huber loss  : robust to real ERCOT price spikes
- Grad clipping: stabilizes training on volatile series
- Chronological split: train 2019-2023 / val 2024 / test 2025-2026

Run after 3_features.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = 'ercot_features.csv'

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

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN      = 48    # hours of history fed to LSTM per sample
PRED_HORIZON = 24    # hours ahead to forecast
BATCH_SIZE   = 128
EPOCHS       = 10
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2


# ── Dataset ───────────────────────────────────────────────────────────────────

class EnergyDataset(Dataset):
    """
    Sliding-window dataset for multi-step price forecasting.
    Each sample: 48 hours of features -> next 24 hours of prices.
    """
    def __init__(self, data: np.ndarray, seq_len: int, horizon: int):
        self.data    = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                              # (seq_len, n_feat)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, 0]  # price only
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────

class PriceLSTM(nn.Module):
    """
    2-layer LSTM for multi-step electricity price forecasting.

    Takes a sequence of hourly feature vectors and outputs
    the next PRED_HORIZON price values in one forward pass.
    """
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 pred_horizon: int = PRED_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, pred_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]       # take final timestep
        return self.fc(self.dropout(last_hidden))


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_dl, val_dl, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.HuberLoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        batch_losses = []
        for batch_idx, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_dl)} "
                      f"| Loss: {loss.item():.5f}", flush=True)

        train_loss = np.mean(batch_losses)

        # Validate
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_batch_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pt')

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.5f} "
              f"| Val: {val_loss:.5f}", flush=True)

    print(f"\nBest val loss: {best_val_loss:.5f}")
    return train_losses, val_losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_dl, price_scaler, device):
    model.load_state_dict(torch.load('best_lstm_model.pt', map_location=device))
    model.eval()

    all_preds, all_actuals = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            all_preds.append(model(xb.to(device)).cpu().numpy())
            all_actuals.append(yb.numpy())

    preds_s   = np.vstack(all_preds)
    actuals_s = np.vstack(all_actuals)

    def inverse(arr):
        return price_scaler.inverse_transform(
            arr.reshape(-1, 1)
        ).reshape(arr.shape)

    preds_inv   = inverse(preds_s)
    actuals_inv = inverse(actuals_s)

    mae  = mean_absolute_error(actuals_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(actuals_inv.flatten(), preds_inv.flatten()))
    mape = np.mean(np.abs(
        (actuals_inv.flatten() - preds_inv.flatten())
        / (np.abs(actuals_inv.flatten()) + 1)
    )) * 100

    return preds_inv, actuals_inv, {'mae': mae, 'rmse': rmse, 'mape': mape}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train Loss', color='#2A7F8F', linewidth=1.5)
    ax.plot(val_losses,   label='Val Loss',   color='#d94f3d', linewidth=1.5)
    ax.set_title('LSTM Training Curve (Huber Loss)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_training_curve.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved lstm_training_curve.png")


def plot_forecasts(preds_inv, actuals_inv):
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    fig.suptitle(
        'LSTM: 24-Hour Ahead Forecasts vs Actual (Test Set 2025-2026)',
        fontsize=13, fontweight='bold'
    )
    sample_indices = np.random.choice(len(preds_inv), 5, replace=False)

    for i, idx in enumerate(sample_indices):
        mae_i = mean_absolute_error(actuals_inv[idx], preds_inv[idx])
        axes[i].plot(range(24), actuals_inv[idx],
                     label='Actual', color='#333333', linewidth=2,
                     marker='o', markersize=3)
        axes[i].plot(range(24), preds_inv[idx],
                     label='LSTM Forecast', color='#2A7F8F', linewidth=2,
                     linestyle='--', marker='s', markersize=3)
        axes[i].set_title(f'Sample {idx} | MAE = ${mae_i:.2f}/MWh')
        axes[i].set_ylabel('$/MWh')
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=30)
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_24h_forecasts.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved lstm_24h_forecasts.png")


def plot_horizon_mae(preds_inv, actuals_inv, overall_mae):
    horizon_maes = [
        mean_absolute_error(actuals_inv[:, h], preds_inv[:, h])
        for h in range(PRED_HORIZON)
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(1, 25), horizon_maes, color='#2A7F8F', alpha=0.85, edgecolor='white')
    ax.axhline(overall_mae, color='#d94f3d', linestyle='--',
               label=f'Overall MAE: ${overall_mae:.2f}')
    ax.set_title('Forecast Error by Horizon Hour (h+1 through h+24)')
    ax.set_xlabel('Hours Ahead'); ax.set_ylabel('MAE ($/MWh)')
    ax.set_xticks(range(1, 25)); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_horizon_mae.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved lstm_horizon_mae.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch {torch.__version__} | device: {device}")

    # Load features
    print(f"\nLoading {INPUT}...")
    df = pd.read_csv(INPUT, parse_dates=['datetime'])
    print(f"Rows: {len(df):,} | {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    # Chronological split happens BEFORE scaling to prevent data leakage.
    # Fitting the scaler on the full dataset would let it "see" the min/max
    # range of 2025-2026 test data while scaling the training set.
    train_mask = df['datetime'] < '2024-01-01'
    val_mask   = (df['datetime'] >= '2024-01-01') & (df['datetime'] < '2025-01-01')
    test_mask  = df['datetime'] >= '2025-01-01'

    n_train = train_mask.sum()
    n_val   = val_mask.sum()

    train_df = df[train_mask]
    val_df   = df[val_mask]
    test_df  = df[test_mask]

    # Fit scaler on training data only, then transform all three splits
    # with that same fitted scaler. Validation and test data never
    # influence the scaling parameters.
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS])

    train_scaled = scaler.transform(train_df[FEATURE_COLS])
    val_scaled   = scaler.transform(val_df[FEATURE_COLS])
    test_scaled  = scaler.transform(test_df[FEATURE_COLS])

    price_scaler = MinMaxScaler()
    price_scaler.fit(train_df[['price_mwh']])

    train_dl = DataLoader(
        EnergyDataset(train_scaled, SEQ_LEN, PRED_HORIZON),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(
        EnergyDataset(val_scaled, SEQ_LEN, PRED_HORIZON),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_dl = DataLoader(
        EnergyDataset(test_scaled, SEQ_LEN, PRED_HORIZON),
        batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"\nTrain: 2019-2023 ({n_train:,} hrs) | Val: 2024 ({n_val:,} hrs) | Test: 2025-2026")
    print(f"Scaler fit on training data only ({n_train:,} hrs)")
    print(f"Batches: train={len(train_dl)} | val={len(val_dl)} | test={len(test_dl)}")

    # Build model
    model = PriceLSTM(input_size=len(FEATURE_COLS)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Train
    print("\nTraining LSTM...")
    train_losses, val_losses = train_model(model, train_dl, val_dl, device)
    plot_training_curve(train_losses, val_losses)

    # Evaluate
    print("\nEvaluating on test set (2025-2026)...")
    preds_inv, actuals_inv, metrics = evaluate_model(model, test_dl, price_scaler, device)

    print("\n" + "=" * 40)
    print("LSTM Results (Test: 2025-2026)")
    print("=" * 40)
    print(f"MAE  : ${metrics['mae']:.2f}/MWh")
    print(f"RMSE : ${metrics['rmse']:.2f}/MWh")
    print(f"MAPE : {metrics['mape']:.1f}%")
    print("=" * 40)

    # Plots
    plot_forecasts(preds_inv, actuals_inv)
    plot_horizon_mae(preds_inv, actuals_inv, metrics['mae'])

    print("\nAll done. Charts saved.")
