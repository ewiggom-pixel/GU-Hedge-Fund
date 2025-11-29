import io, pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

PORT_FILE = "10_Industry_Portfolios_Daily.csv"  
FF5_FILE  = "F-F_Research_Data_5_Factors_2x3_daily.csv"    
PORT_SERIES = "HiTec"       

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
        first_num = next(i for i, ln in enumerate(lines)
                         if ln.strip().startswith(tuple(str(y) for y in range(1920, 2030))) and "," in ln)
        hdr_idx = max(0, first_num - 1)

    end_idx = next((j for j in range(hdr_idx+1, len(lines))
                    if lines[j].strip() == "" or lines[j].strip().lower().startswith("annual")), len(lines))
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

    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df

# Load FF5 daily factors (percent -> decimal)
ff5_lines = read_first_text(FF5_FILE)
ff5 = extract_daily_table(ff5_lines)
ff5.columns = [c.replace("-", "_") for c in ff5.columns]
for c in ff5.columns:
    if c != "Date":
        ff5[c] = pd.to_numeric(ff5[c], errors="coerce")/100.0

# Load the chosen portfolio (percent -> decimal)
port_lines = read_first_text(PORT_FILE)
port = extract_daily_table(port_lines)
# Convert all non-Date columns to decimals
for c in port.columns:
    if c != "Date":
        port[c] = pd.to_numeric(port[c], errors="coerce")/100.0

# Verify the chosen series exists
if PORT_SERIES not in port.columns:
    raise ValueError(f"'{PORT_SERIES}' not found. Available columns: {', '.join([c for c in port.columns if c!='Date'][:30])} ...")

# Build dependent variable from chosen portfolio column
dep = port[["Date", PORT_SERIES]].rename(columns={PORT_SERIES: "Portfolio_Return"}).dropna()

# Merge & compute excess return
df = dep.merge(ff5, on="Date", how="inner").dropna()
df["Excess_Ret"] = df["Portfolio_Return"] - df["RF"]

# OLS regression: Excess_Ret ~ Mkt_RF + SMB + HML + RMW + CMA
X = sm.add_constant(df[["Mkt_RF","SMB","HML","RMW","CMA"]])
y = df["Excess_Ret"]
model = sm.OLS(y, X).fit()

# Print concise results
params = model.params.rename({
    "const":"Alpha","Mkt_RF":"Beta_Mkt","SMB":"Beta_SMB","HML":"Beta_HML","RMW":"Beta_RMW","CMA":"Beta_CMA"
}).round(4)
print(f"Obs: {int(model.nobs)}  R^2: {model.rsquared:.4f}")
print(params.to_frame("Coefficient"))

# -------------------------------------------------------
# Compute rolling betas (use 60-day window, can adjust)
# -------------------------------------------------------
WINDOW = 60
beta_rows = []

factor_cols = ["Mkt_RF","SMB","HML","RMW","CMA"]

for i in range(WINDOW, len(df)):
    window = df.iloc[i-WINDOW:i]

    y_win = window["Excess_Ret"]
    X_win = sm.add_constant(window[factor_cols])

    model_win = sm.OLS(y_win, X_win).fit()

    date = df.iloc[i]["Date"]
    params = model_win.params

    beta_rows.append([
        date,
        params.get("Mkt_RF", np.nan),
        params.get("SMB", np.nan),
        params.get("HML", np.nan),
        params.get("RMW", np.nan),
        params.get("CMA", np.nan)
    ])

rolling_betas = pd.DataFrame(
    beta_rows,
    columns=["Date","beta_Mkt","beta_SMB","beta_HML","beta_RMW","beta_CMA"]
)

rolling_betas.to_csv("rolling_betas.csv", index=False)
print("Saved rolling betas → rolling_betas.csv")
