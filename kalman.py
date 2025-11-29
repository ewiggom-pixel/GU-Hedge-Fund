import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------
# Load WRDS FF5 data
# ---------------------------------------------------
df = pd.read_csv(
    "F-F_Research_Data_5_Factors_2x3_daily.csv",
    skiprows=3,
    skipfooter=1,
    engine="python"
)
df = df.rename(columns={df.columns[0]: "Date"})
df.columns = [c.replace("-", "_").replace(" ", "") for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
df = df.dropna(subset=["Date"])

factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
for c in factor_cols + ["RF"]:
    df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # percent → decimal

# ---------------------------------------------------
# Load beta file
# ---------------------------------------------------
betas = pd.read_csv("rolling_betas.csv")
betas["Date"] = pd.to_datetime(betas["Date"])
beta_cols = ["beta_Mkt", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA"]

# Merge factors and betas
data = df.merge(betas, on="Date").dropna(subset=factor_cols + beta_cols)

# ---------------------------------------------------
# Normalize factors and betas
# ---------------------------------------------------
factor_mean = data[factor_cols].mean()
factor_std = data[factor_cols].std()
state_norm = (data[factor_cols] - factor_mean) / factor_std

beta_mean = data[beta_cols].mean()
beta_std = data[beta_cols].std()
obs_norm = (data[beta_cols] - beta_mean) / beta_std

states = state_norm.values
observations = obs_norm.values

n_states = len(factor_cols)
n_obs = len(beta_cols)

# Kalman filter matrices
A = np.eye(n_states)         # state transition
H = np.eye(n_obs, n_states)  # observation matrix
Q = np.eye(n_states) * 1e-5  # process noise
R = np.eye(n_obs) * 1e-3     # observation noise

kf = KalmanFilter(
    transition_matrices=A,
    observation_matrices=H,
    transition_covariance=Q,
    observation_covariance=R,
    initial_state_mean=states[0],
    initial_state_covariance=np.eye(n_states)
)

# Run filter
filtered_means, filtered_covs = kf.filter(observations)

# ---------------------------------------------------
# Predict next-day factors (normalized)
# ---------------------------------------------------
next_state_norm, _ = kf.filter_update(
    filtered_means[-1],
    filtered_covs[-1],
    observation=None
)

# Convert normalized prediction back to original units
next_state = next_state_norm * factor_std.values + factor_mean.values

print("=== Predicted Next-Day Factors ===")
for c, v in zip(factor_cols, next_state):
    print(f"{c}: {v:.6f}")

# ---------------------------------------------------
# One-step-ahead evaluation
# ---------------------------------------------------
one_step_preds_norm = filtered_means[:-1] @ A.T
true_next_norm = states[1:]

# Convert normalized predictions back to original units
one_step_preds = one_step_preds_norm * factor_std.values + factor_mean.values
true_next = true_next_norm * factor_std.values + factor_mean.values

# Per-factor metrics
results = []
for j, name in enumerate(factor_cols):
    y_true = true_next[:, j]
    y_pred = one_step_preds[:, j]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    sign_true = np.sign(y_true); sign_true[sign_true==0]=1
    sign_pred = np.sign(y_pred); sign_pred[sign_pred==0]=1
    dir_acc = (sign_true == sign_pred).mean()

    results.append({
        "factor": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "dir_acc": dir_acc
    })

metrics_df = pd.DataFrame(results).set_index("factor")

# Overall metrics across all factors
mse_all = mean_squared_error(true_next.ravel(), one_step_preds.ravel())
rmse_all = np.sqrt(mse_all)

# 95% predictive interval coverage (original units)
inside_counts = np.zeros(len(factor_cols))
total_counts = 0
for t in range(filtered_means.shape[0]-1):
    pred_mean_norm = A @ filtered_means[t]
    pred_cov_norm = A @ filtered_covs[t] @ A.T + Q
    pred_mean = pred_mean_norm * factor_std.values + factor_mean.values
    se = np.sqrt(np.maximum(np.diag(pred_cov_norm),0)) * factor_std.values
    lower = pred_mean - 1.96*se
    upper = pred_mean + 1.96*se
    true_val = true_next[t]
    inside_counts += ((true_val >= lower) & (true_val <= upper)).astype(int)
    total_counts += 1
coverage = inside_counts / total_counts

# Print evaluation
print("\n=== Kalman Filter One-Step-Ahead Evaluation ===")
print(f"Samples evaluated: {one_step_preds.shape[0]}")
print(f"Overall RMSE (all factors): {rmse_all:.6e}  MSE: {mse_all:.6e}\n")
print("Per-factor metrics:")
print(metrics_df.round(6).to_string())
print("\n95% predictive interval coverage per factor:")
for name, cov in zip(factor_cols, coverage):
    print(f"  {name}: {cov*100:.2f}%")
