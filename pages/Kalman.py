import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import statsmodels.api as sm
from dlm import em_dlm_multifactor

st.set_page_config(
    page_title="Dynamic Multi-Factor Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
   background: linear-gradient(180deg, #000000 0%, #072f5f 100%);
}

.stTabs [data-baseweb="tab-list"] button {
    color: #cbeaff !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #58cced !important;
    border-bottom: 3px solid #3895d3 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #3895d3 !important;
}

.hero h1 {
    font-size: 3.0rem;
    text-align: center;
    margin: 2rem 0 0.75rem;
    color: #cbeaff;
}
.hero p {
    font-size: 1.1rem;
    color: #ccc;
    text-align: center;
    margin-bottom: 2rem;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

.dashboard-btn {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.dashboard-btn a {
    background: linear-gradient(90deg, #3895d3, #58cced);
    border-radius: 999px;
    color: white !important;
    padding: 0.75rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: opacity 0.2s ease, transform 0.15s ease-in-out;
}
.dashboard-btn a:hover {
    opacity: 0.85;
    transform: scale(1.05);
}

[data-testid="stTextInput"] input,
[data-testid="stPassword"] input {
    background-color: #000814;
    color: #cbeaff;
    border-radius: 0.75rem;
    border: 1px solid #001428;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stPassword"] input:focus {
    border-color: #3895d3;
    box-shadow: 0 0 0 1px #3895d3;
    outline: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Dynamic Multi-Factor Risk & Alpha</h1>
  <p>This page uses the DLM/Kalman model fitted on the regression page to show time-varying factor
     exposures, risk decomposition, and expected excess returns.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SESSION + CORE OBJECTS -----------------

if (
    "dlm_result" not in st.session_state
    or "dlm_df" not in st.session_state
    or "dlm_factor_cols" not in st.session_state
):
    st.error("No DLM results found. Please run the FF5 + Momentum Regression page first.")
    st.stop()

res_dlm = st.session_state["dlm_result"]
df = st.session_state["dlm_df"].copy()
factor_cols = st.session_state["dlm_factor_cols"]
periods_per_year = st.session_state.get("dlm_periods_per_year", 252)
freq_label = st.session_state.get("dlm_freq_label", "Daily")

y = df["excess_ret"].to_numpy(dtype=float)
F = df[factor_cols].to_numpy(dtype=float)

# ----------------- DLM SUMMARY -----------------

alpha_full = float(res_dlm["alpha"])
R_full = float(res_dlm["R"])
beta_smooth = np.asarray(res_dlm["beta_smooth"], dtype=float)

alpha_period = alpha_full
alpha_annual = (1 + alpha_period) ** periods_per_year - 1

if freq_label == "Daily":
    days_week = 5
    days_month = 21
    alpha_1 = alpha_period
    alpha_week = (1 + alpha_period) ** days_week - 1
    alpha_month = (1 + alpha_period) ** days_month - 1
else:
    alpha_1m = alpha_period
    alpha_3m = (1 + alpha_period) ** 3 - 1
    alpha_12m = (1 + alpha_period) ** periods_per_year - 1

# ----------------- TIME-VARYING BETAS PREP -----------------

beta_df = pd.DataFrame(beta_smooth, columns=factor_cols)
beta_df["date"] = df["date"].values
beta_df = beta_df.set_index("date")
df_idx = df.set_index("date")

# ----------------- RISK / VARIANCE DECOMP PREP -----------------

Sigma_f = np.cov(F, rowvar=False)
beta_T = beta_smooth[-1]
var_factor = float(beta_T @ Sigma_f @ beta_T)
var_idio = R_full
var_total = var_factor + var_idio

factor_contrib = beta_T * (Sigma_f @ beta_T)
factor_contrib = np.asarray(factor_contrib, dtype=float)
if var_factor > 0:
    pct_contrib = factor_contrib / var_factor
else:
    pct_contrib = np.zeros_like(factor_contrib)

var_df = pd.DataFrame(
    {
        "factor": factor_cols,
        "variance_contribution": factor_contrib,
        "pct_of_factor_var": pct_contrib,
    }
)

# ----------------- FACTOR COLOR PALETTE -----------------

factor_color_map = {
    "mktrf": "#3895d3",
    "smb": "#58cced",
    "hml": "#1261a0",
    "rmw": "#072f5f",
    "cma": "#001329",
    "umd": "#ecf7ff",
}

palette_for_factors = [factor_color_map.get(f, "#3895d3") for f in factor_cols]

var_df["pct_of_factor_var"] = var_df["pct_of_factor_var"].fillna(0.0)

# ----------------- DONUT CHART -----------------

if var_factor > 0 and np.all(np.isfinite(var_df["variance_contribution"])):
    base = alt.Chart(var_df)

    arcs = (
        base
        .mark_arc(outerRadius=120, innerRadius=50)
        .encode(
            theta=alt.Theta("pct_of_factor_var:Q", stack=True),
            color=alt.Color(
                "factor:N",
                scale=alt.Scale(domain=factor_cols, range=palette_for_factors),
                legend=alt.Legend(title="factor")
            ),
            tooltip=[
                alt.Tooltip("factor:N", title="Factor"),
                alt.Tooltip("variance_contribution:Q",
                            title="Variance", format=".3e"),
                alt.Tooltip("pct_of_factor_var:Q",
                            title="% of factor variance", format=".1%"),
            ],
        )
    )

    pie = arcs.properties(
        height=350,
        title="Factor Contribution to Factor Variance (%)"
    )
else:
    pie = None

# ----------------- TABS LAYOUT -----------------

tab_alpha, tab_betas, tab_risk, tab_backtest = st.tabs(
    ["Alpha & Expected Return", "Time-Varying Betas", "Risk & Variance Decomposition", "Backtest & Validation"]
)

# --- TAB 1: ALPHA & EXPECTED RETURN ---
with tab_alpha:
    st.subheader(f"DLM Alpha and Expected {freq_label} Excess Return")

    if freq_label == "Daily":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Alpha (per day)", f"{alpha_1:.6f}")
        with c2:
            st.metric("Expected excess (1 week)", f"{alpha_week:.4%}")
        with c3:
            st.metric("Expected excess (1 month)", f"{alpha_month:.4%}")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Expected excess (1 month)", f"{alpha_1m:.4%}")
        with c2:
            st.metric("Expected excess (3 months)", f"{alpha_3m:.4%}")
        with c3:
            st.metric("Expected excess (1 year)", f"{alpha_12m:.4%}")

    st.markdown("**Annualized alpha from DLM:**")
    st.write(f"{alpha_annual:.4%}")

# --- TAB 2: TIME-VARYING BETAS ---
with tab_betas:
    st.subheader("Time-Varying Factor Exposures: DLM vs Rolling OLS (Kalman)")

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
            st.markdown(
                f"**{name} beta: DLM (state-space) vs Rolling OLS + 1D Kalman**"
            )
            st.line_chart(overlay)
        else:
            st.markdown(f"**{name} beta (DLM only)**")
            st.line_chart(beta_df[[name]])

# --- TAB 3: RISK & VARIANCE DECOMPOSITION ---
with tab_risk:
    st.subheader("Risk Decomposition At Most Recent Date")

    row1, row2, row3 = st.columns(3)
    with row1:
        st.metric(f"Total {freq_label.lower()} variance", f"{var_total:.6e}")
    with row2:
        st.metric(f"Factor variance ({freq_label.lower()})", f"{var_factor:.6e}")
    with row3:
        st.metric("Idiosyncratic variance (R)", f"{var_idio:.6e}")

    st.subheader("Factor Variance Contribution")
    if pie is not None:
        st.altair_chart(pie, use_container_width=True)
    else:
        st.info(
            "Factor variance is zero or invalid; cannot plot factor contribution pie chart."
        )

    # ---------- NEW TABLES ----------
    st.subheader("Table: % of Factor Variance by Factor")
    factor_table = var_df.copy()
    factor_table = factor_table[["factor", "variance_contribution", "pct_of_factor_var"]]
    factor_table = factor_table.rename(
        columns={
            "factor": "Factor",
            "variance_contribution": "Factor variance",
            "pct_of_factor_var": "% of factor variance",
        }
    )
    st.dataframe(
        factor_table.style.format(
            {
                "Factor variance": "{:.3e}",
                "% of factor variance": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Table: Factor vs Idiosyncratic Variance (% of total)")
    if var_total > 0:
        pct_factor_total = var_factor / var_total
        pct_idio_total = var_idio / var_total
    else:
        pct_factor_total = np.nan
        pct_idio_total = np.nan

    comp_table = pd.DataFrame(
        {
            "Component": ["Factor variance", "Idiosyncratic variance"],
            "Variance": [var_factor, var_idio],
            "% of total variance": [pct_factor_total, pct_idio_total],
        }
    )
    st.dataframe(
        comp_table.style.format(
            {
                "Variance": "{:.3e}",
                "% of total variance": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

# --- TAB 4: BACKTEST & VALIDATION ---
with tab_backtest:
    st.subheader("Backtest: DLM one-step-ahead projections")

    if len(df) < 40:
        st.info("Not enough observations to run a meaningful backtest.")
    else:
        backtest_type = st.radio(
            "Backtest model:",
            ["DLM 1-step-ahead (EM on train only)", "Rolling re-estimated DLM"],
        )

        train_frac = st.slider(
            "Training fraction (earliest data used for model estimation)",
            0.5,
            0.9,
            0.7,
            0.05,
        )

        n = len(df)
        n_train = int(n * train_frac)
        dates = df["date"].to_numpy()

        y_train = y[:n_train]
        F_train = F[:n_train]
        y_test = y[n_train:]

        def _backtest_metrics(y_true, y_pred):
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                return np.nan, np.nan, np.nan
            yt = y_true[mask]
            yp = y_pred[mask]
            mse = float(np.mean((yp - yt) ** 2))
            sst = float(np.sum((yt - yt.mean()) ** 2))
            r2 = float(1.0 - np.sum((yp - yt) ** 2) / sst) if sst > 0 else np.nan
            hit = float(np.mean(np.sign(yt) == np.sign(yp)))
            return mse, r2, hit

        if backtest_type == "DLM 1-step-ahead (EM on train only)":
            res_train = em_dlm_multifactor(y_train, F_train)
            alpha_tr = float(res_train["alpha"])
            Q_tr = np.asarray(res_train["Q"], dtype=float)
            R_tr = float(res_train["R"])
            beta_filt_tr = np.asarray(res_train["beta_filt"], dtype=float)
            P_filt_tr = np.asarray(res_train["P_filt"], dtype=float)

            m_prev = beta_filt_tr[-1].copy()
            P_prev = P_filt_tr[-1].copy()

            yhat_test = np.full(len(y_test), np.nan)

            for i, t in enumerate(range(n_train, n)):
                ft = F[t]

                m_pred = m_prev
                P_pred = P_prev + Q_tr

                yhat_t = alpha_tr + ft @ m_pred
                yhat_test[i] = yhat_t

                innov = y[t] - yhat_t
                S = float(ft @ P_pred @ ft + R_tr)
                if S <= 0 or not np.isfinite(S):
                    m_filt_t = m_pred
                    P_filt_t = P_pred
                else:
                    K_t = (P_pred @ ft) / S
                    m_filt_t = m_pred + K_t * innov
                    P_filt_t = P_pred - np.outer(K_t, ft) @ P_pred

                m_prev = m_filt_t
                P_prev = P_filt_t

            mse, r2, hit = _backtest_metrics(y_test, yhat_test)
            model_name = "DLM 1-step-ahead (train EM only)"

        else:
            if freq_label == "Daily":
                window = 252
            else:
                window = 36

            st.info(
                f"Rolling EM window: {window} observations "
                f"({freq_label.lower()})"
            )

            yhat_test = np.full(len(y_test), np.nan)

            for i, t in enumerate(range(n_train, n)):
                start = max(0, t - window)
                end = t
                y_win = y[start:end]
                F_win = F[start:end]

                res_win = em_dlm_multifactor(y_win, F_win)
                alpha_win = float(res_win["alpha"])
                beta_filt_win = np.asarray(res_win["beta_filt"], dtype=float)

                beta_last = beta_filt_win[-1]
                yhat_t = alpha_win + F[t] @ beta_last
                yhat_test[i] = yhat_t

            mse, r2, hit = _backtest_metrics(y_test, yhat_test)
            model_name = "Rolling re-estimated DLM"

        rng = np.random.default_rng(0)

        baseline_zero = np.zeros_like(y_test)
        mse_zero, r2_zero, hit_zero = _backtest_metrics(y_test, baseline_zero)

        mean_train = float(np.mean(y_train))
        baseline_mean = np.full_like(y_test, mean_train)
        mse_mean, r2_mean, hit_mean = _backtest_metrics(y_test, baseline_mean)

        idx = rng.permutation(len(y_test))
        y_test_shuffled = y_test[idx]
        mse_shuff, r2_shuff, hit_shuff = _backtest_metrics(y_test_shuffled, yhat_test)

        results = pd.DataFrame(
            {
                "Model": [model_name],
                "Test MSE": [mse],
                "Test R²": [r2],
                "Direction hit-rate": [hit],
            }
        )

        st.subheader("Backtest Performance")
        st.dataframe(
            results.style.format(
                {
                    "Test MSE": "{:.4e}",
                    "Test R²": "{:.3f}",
                    "Direction hit-rate": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        sanity = pd.DataFrame(
            {
                "Model": [
                    model_name,
                    "Baseline: predict 0",
                    "Baseline: predict train mean",
                    "Shuffled y_test vs DLM",
                ],
                "Test MSE": [mse, mse_zero, mse_mean, mse_shuff],
                "Test R²": [r2, r2_zero, r2_mean, r2_shuff],
                "Direction hit-rate": [hit, hit_zero, hit_mean, hit_shuff],
            }
        )

        st.subheader("Sanity Checks")
        st.dataframe(
            sanity.style.format(
                {
                    "Test MSE": "{:.4e}",
                    "Test R²": "{:.3f}",
                    "Direction hit-rate": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        test_len = len(y_test)
        frac_pos = float(np.mean(y_test > 0))
        std_y = float(np.std(y_test))

        st.subheader("Test Sample Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Number of test observations", f"{test_len}")
        with c2:
            st.metric("Fraction positive (y_test > 0)", f"{frac_pos:.1%}")
        with c3:
            st.metric("Std dev of y_test", f"{std_y:.4f}")

        back_df = pd.DataFrame(
            {
                "date": dates[n_train:],
                "Actual excess return": y_test,
                "Predicted excess return": yhat_test,
            }
        ).set_index("date")

        st.subheader("Test-Period Actual vs Predicted Excess Returns")
        st.line_chart(back_df)

# Back button
st.markdown("""
<div class="dashboard-btn">
    <a href="/CSV" target="_self">Back to Regression</a>
</div>
""", unsafe_allow_html=True)
