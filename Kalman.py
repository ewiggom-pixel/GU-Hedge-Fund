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

y = df["excess_ret"].to_numpy(dtype=float)
F = df[factor_cols].to_numpy(dtype=float)

alpha = float(res_dlm["alpha"])
Q = np.asarray(res_dlm["Q"], dtype=float)
R = float(res_dlm["R"])
beta0 = np.asarray(res_dlm["beta0"], dtype=float)
P0 = np.asarray(res_dlm["P0"], dtype=float)
beta_smooth = np.asarray(res_dlm["beta_smooth"], dtype=float)

st.subheader("Time-Varying Factor Exposures (Smoothed Betas)")

beta_df = pd.DataFrame(beta_smooth, columns=factor_cols)
beta_df["date"] = df["date"].values
beta_plot = beta_df.set_index("date")[factor_cols]
st.line_chart(beta_plot)

st.subheader("Risk Decomposition At Most Recent Date")

Sigma_f = np.cov(F, rowvar=False)
beta_T = beta_smooth[-1]
var_factor = float(beta_T @ Sigma_f @ beta_T)
var_idio = R
var_total = var_factor + var_idio

row1, row2, row3 = st.columns(3)
with row1:
    st.metric("Total daily variance", f"{var_total:.6e}")
with row2:
    st.metric("Factor variance", f"{var_factor:.6e}")
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

st.subheader("Expected Excess Return From DLM Alpha")

alpha_daily = alpha
days_week = 5
days_month = 21
days_year = 252

alpha_week = (1 + alpha_daily) ** days_week - 1
alpha_month = (1 + alpha_daily) ** days_month - 1
alpha_annual = (1 + alpha_daily) ** days_year - 1

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("E[excess] (1 day)", f"{alpha_daily:.6f}")
with c2:
    st.metric("E[excess] (1 week)", f"{alpha_week:.4%}")
with c3:
    st.metric("E[excess] (1 month)", f"{alpha_month:.4%}")
with c4:
    st.metric("E[excess] (1 year)", f"{alpha_annual:.4%}")
