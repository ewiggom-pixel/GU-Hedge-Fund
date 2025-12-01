import numpy as np
import pandas as pd
import streamlit as st

st.title("Dynamic Multi-Factor: Time-Varying Betas, Risk, And Validation")

st.write(
    "This page uses the dynamic multi-factor model (DLM/Kalman) estimated on the "
    "FF5 + Momentum Regression page to show time-varying factor exposures, risk "
    "decomposition, and expected excess returns."
)

if (
    "dlm_result" not in st.session_state
    or "dlm_df" not in st.session_state
    or "dlm_factor_cols" not in st.session_state
):
    st.error("No DLM results found. Run the FF5 + Momentum Regression page first.")
    st.stop()

res_dlm = st.session_state["dlm_result"]
df = st.session_state["dlm_df"].copy()
factor_cols = st.session_state["dlm_factor_cols"]
periods_per_year = st.session_state.get("dlm_periods_per_year", 252)
freq_label = st.session_state.get("dlm_freq_label", "Daily")

y = df["excess_ret"].to_numpy(dtype=float)
F = df[factor_cols].to_numpy(dtype=float)

alpha = float(res_dlm["alpha"])
R = float(res_dlm["R"])
beta_smooth = np.asarray(res_dlm["beta_smooth"], dtype=float)

# --- Alpha & expected returns ---
st.subheader(f"DLM Alpha and Expected {freq_label} Excess Return")

alpha_period = alpha  # per day or per month depending on freq
alpha_annual = (1 + alpha_period) ** periods_per_year - 1

if freq_label == "Daily":
    # daily: 1 day, 1 week, 1 month equivalents
    days_week = 5
    days_month = 21
    alpha_1 = (1 + alpha_period) ** 1 - 1
    alpha_week = (1 + alpha_period) ** days_week - 1
    alpha_month = (1 + alpha_period) ** days_month - 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("E[excess] (1 day)", f"{alpha_1:.6f}")
    with c2:
        st.metric("E[excess] (1 week)", f"{alpha_week:.4%}")
    with c3:
        st.metric("E[excess] (1 month)", f"{alpha_month:.4%}")

else:  # Monthly
    # monthly: 1m, 3m, 12m
    alpha_1m = (1 + alpha_period) ** 1 - 1
    alpha_3m = (1 + alpha_period) ** 3 - 1
    alpha_12m = (1 + alpha_period) ** periods_per_year - 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("E[excess] (1 month)", f"{alpha_1m:.4%}")
    with c2:
        st.metric("E[excess] (3 months)", f"{alpha_3m:.4%}")
    with c3:
        st.metric("E[excess] (12 months)", f"{alpha_12m:.4%}")

st.markdown(
    f"**Alpha ({freq_label.lower()})**: {alpha_period:.6g}  "
    f"(**annualized** ≈ {alpha_annual:.2%})"
)

# --- Time-varying exposures: DLM vs rolling OLS (Kalman) ---
st.subheader("Time-Varying Factor Exposures: DLM vs Rolling OLS (Kalman)")

beta_df = pd.DataFrame(beta_smooth, columns=factor_cols)
beta_df["date"] = df["date"].values
beta_df = beta_df.set_index("date")
df_idx = df.set_index("date")

for name in factor_cols:
    col_k = f"{name}_beta_kalman"
    if col_k in df_idx.columns:
        overlay = pd.concat(
            [
                beta_df[[name]].rename(columns={name: "DLM (state-space) beta"}),
                df_idx[[col_k]].rename(
                    columns={col_k: "Rolling OLS + 1D Kalman beta"}
                ),
            ],
            axis=1,
        )
        st.markdown(f"**{name} beta: DLM (state-space) vs Rolling OLS + 1D Kalman**")
        st.line_chart(overlay)
    else:
        st.markdown(f"**{name} beta (DLM only)**")
        st.line_chart(beta_df[[name]])

# --- Risk decomposition ---
st.subheader("Risk Decomposition At Most Recent Date")

Sigma_f = np.cov(F, rowvar=False)
beta_T = beta_smooth[-1]
var_factor = float(beta_T @ Sigma_f @ beta_T)
var_idio = R
var_total = var_factor + var_idio

row1, row2, row3 = st.columns(3)
with row1:
    st.metric(f"Total {freq_label.lower()} variance", f"{var_total:.6e}")
with row2:
    st.metric(f"Factor variance ({freq_label.lower()})", f"{var_factor:.6e}")
with row3:
    st.metric("Idiosyncratic variance (R)", f"{var_idio:.6e}")

contrib = beta_T * (Sigma_f @ beta_T)
contrib_df = pd.DataFrame(
    {
        "factor": factor_cols,
        "variance_contribution": contrib,
    }
)
if var_factor > 0:
    contrib_df["pct_of_factor_var"] = contrib_df["variance_contribution"] / var_factor
else:
    contrib_df["pct_of_factor_var"] = np.nan

st.write("Factor contribution to variance")
st.dataframe(
    contrib_df.style.format(
        {"variance_contribution": "{:.3e}", "pct_of_factor_var": "{:.2%}"}
    )
)
