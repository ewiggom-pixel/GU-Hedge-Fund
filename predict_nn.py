import io, pandas as pd, numpy as np
import statsmodels.api as sm
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


PORT_FILE = "10_Industry_Portfolios_Daily.csv"
FF5_FILE  = "F-F_Research_Data_5_Factors_2x3_daily.csv"
PORT_SERIES = "HiTec"
WINDOW = 252

def read_first_text(file_path: str) -> list:
    with open(file_path, "r", encoding="latin1") as f:
        text = f.read()
    return text.splitlines()

def extract_daily_table(lines: list) -> pd.DataFrame:
    hdr_idx = None
    for i, ln in enumerate(lines):
        s = ln.lower().replace(" ", "")
        if ("mkt" in s and "rf" in s) or s.startswith("date,") or s.startswith("date "):
            hdr_idx = i; break
    if hdr_idx is None:
        first_num = next(
            i for i, ln in enumerate(lines)
            if ln.strip().startswith(tuple(str(y) for y in range(1920, 2030))) and "," in ln
        )
        hdr_idx = max(0, first_num - 1)

    end_idx = next(
        (j for j in range(hdr_idx + 1, len(lines))
         if lines[j].strip() == "" or lines[j].strip().lower().startswith("annual")),
        len(lines)
    )
    block = "\n".join(lines[hdr_idx:end_idx])

    try:
        df = pd.read_csv(io.StringIO(block), skipinitialspace=True, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(block), delim_whitespace=True, engine="python")

    df.columns = [str(c).strip().replace("-", "_").replace(" ", "") for c in df.columns]
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = df["Date"].astype(str).str.strip()
    df = df[df["Date"].str.match(r"^\d{8}$", na=False)].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    # convert percent → decimal
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df

#load data

ff5_lines = read_first_text(FF5_FILE)
ff5 = extract_daily_table(ff5_lines)
ff5.columns = [c.replace("-", "_") for c in ff5.columns]

port_lines = read_first_text(PORT_FILE)
port = extract_daily_table(port_lines)

if PORT_SERIES not in port.columns:
    raise ValueError(f"'{PORT_SERIES}' not found in portfolio file.")

# Merge & compute excess return
dep = port[["Date", PORT_SERIES]].rename(columns={PORT_SERIES: "Portfolio_Return"}).dropna()
df = dep.merge(ff5, on="Date", how="inner").dropna()
df["Excess_Ret"] = df["Portfolio_Return"] - df["RF"]


#train-test split
feature_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
X_np = df[feature_cols].values.astype("float32")
y_np = df["Excess_Ret"].values.astype("float32").reshape(-1, 1)

print("input shape:", X_np.shape)
print("output shape:", y_np.shape)

split_idx = int(len(df) * 0.8)

X_train, X_test = X_np[:split_idx], X_np[split_idx:]
y_train, y_test = y_np[:split_idx], y_np[split_idx:]
print(f"train samples: {len(X_train)}  test samples: {len(X_test)}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)

X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)


features_list = []
targets_list = []

for start in range(len(df) - WINDOW):
# for start in range(len(df) - WINDOW - 5):
    window_df = df.iloc[start:start+WINDOW]

    # Run regression
    X = sm.add_constant(window_df[["Mkt_RF","SMB","HML","RMW","CMA"]])
    y = window_df["Excess_Ret"]
    model = sm.OLS(y, X).fit()

    # 6 input features
    features = [
        model.params["const"],
        model.params["Mkt_RF"],
        model.params["SMB"],
        model.params["HML"],
        model.params["RMW"],
        model.params["CMA"]
    ]
    features_list.append(features)

    # Next-week excess return is the prediction target
    targets_list.append(df["Excess_Ret"].iloc[start + WINDOW + 5])

# Convert to arrays
X_array = np.array(features_list, dtype=np.float32)
y_array = np.array(targets_list, dtype=np.float32).reshape(-1, 1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

split_idx = int(len(X_scaled) * 0.8)

X_train = torch.tensor(X_scaled[:split_idx], dtype=torch.float32)
y_train = torch.tensor(y_array[:split_idx], dtype=torch.float32)

X_test  = torch.tensor(X_scaled[split_idx:], dtype=torch.float32)
y_test  = torch.tensor(y_array[split_idx:], dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

#neural network initialization
class FF5Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

net = FF5Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(200):
    net.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = net(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    net.eval()
    with torch.no_grad():
        test_pred = net(X_test)
        test_loss = criterion(test_pred, y_test).item()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1}/200   test loss: {test_loss:.6f}")


with torch.no_grad():
    pred = net(X_test).numpy()

print("\nFirst 5 predictions vs actual (TEST SET):")
print(np.hstack([pred[:5], y_test.numpy()[:5]]))