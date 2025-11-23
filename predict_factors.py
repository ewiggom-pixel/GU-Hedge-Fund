import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------
# 1. LOAD FF5 DAILY FACTORS (CSV)
# -----------------------------------------------------
FF5_FILE = "F-F_Research_Data_5_Factors_2x3_daily.csv"

def read_first_text(file_path: str):
    with open(file_path, "r", encoding="latin1") as f:
        return f.read().splitlines()

def extract_daily_table(lines: list) -> pd.DataFrame:
    # Find header
    hdr_idx = None
    for i, ln in enumerate(lines):
        s = ln.lower().replace(" ", "")
        if ("mkt" in s and "rf" in s):
            hdr_idx = i; break
    if hdr_idx is None:
        raise ValueError("Could not locate table header in FF5 file.")

    # Cut until next blank or 'annual'
    end_idx = next((j for j in range(hdr_idx+1, len(lines))
                    if lines[j].strip() == "" or "annual" in lines[j].lower()), len(lines))

    block = "\n".join(lines[hdr_idx:end_idx])

    df = pd.read_csv(io.StringIO(block), engine="python")
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df.dropna()


# Load factors
ff5_lines = read_first_text(FF5_FILE)
df = extract_daily_table(ff5_lines)

factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
df = df[["Date"] + factor_cols]
df = df.sort_values("Date").reset_index(drop=True)

print(df.head())


# -----------------------------------------------------
# 2. CREATE SUPERVISED LEARNING WINDOW DATASET
# -----------------------------------------------------
WINDOW = 20  # 20-day input window

features = []
targets = []

for start in range(len(df) - WINDOW - 1):
    window_slice = df[factor_cols].iloc[start : start + WINDOW].values  # (20 × 5)
    next_day = df[factor_cols].iloc[start + WINDOW].values              # (5)

    features.append(window_slice.flatten())  # flatten to 100-dim
    targets.append(next_day)

features = np.array(features)
targets = np.array(targets)

print("Feature shape:", features.shape)  # (#samples, 100)
print("Target shape:", targets.shape)    # (#samples, 5)


# -----------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------------------------------
train_size = int(len(features) * 0.8)

X_train = features[:train_size]
y_train = targets[:train_size]
X_test  = features[train_size:]
y_test  = targets[train_size:]


# -----------------------------------------------------
# 4. NORMALIZATION
# -----------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)


# -----------------------------------------------------
# 5. CONVERT TO TORCH DATASETS
# -----------------------------------------------------
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)


# -----------------------------------------------------
# 6. DEFINE MODEL
# -----------------------------------------------------
class FactorPredictor(nn.Module):
    def __init__(self, input_dim=WINDOW * 5, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = FactorPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


# -----------------------------------------------------
# 7. TRAINING LOOP
# -----------------------------------------------------
EPOCHS = 40

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Test loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            test_loss += loss_fn(pred, yb).item()

    print(f"Epoch {epoch+1:02d} | Train Loss={train_loss/len(train_loader):.6f} | Test Loss={test_loss/len(test_loader):.6f}")


# -----------------------------------------------------
# 8. PREDICTION DEMO
# -----------------------------------------------------
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t[:10])
    pred = scaler_y.inverse_transform(pred_scaled)
    actual = y_test[:10]

print("\nFirst 10 predictions vs actual:")
for p, a in zip(pred, actual):
    print(f"Pred: {p}   Actual: {a}")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t)
    pred = scaler_y.inverse_transform(pred_scaled)
    actual = y_test

# Compute metrics
mse = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, pred)
r2 = r2_score(actual, pred)

print("\n=== Overall Metrics ===")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.4f}")

# Per-factor metrics
print("\n=== Per-Factor Metrics ===")
for i, factor in enumerate(factor_cols):
    mse_f = mean_squared_error(actual[:, i], pred[:, i])
    rmse_f = np.sqrt(mse_f)
    mae_f = mean_absolute_error(actual[:, i], pred[:, i])
    r2_f = r2_score(actual[:, i], pred[:, i])
    print(f"{factor}: MSE={mse_f:.6f}  RMSE={rmse_f:.6f}  MAE={mae_f:.6f}  R²={r2_f:.4f}")

