import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import wrds
from dlm import em_dlm_multifactor


st.set_page_config(
    page_title="FF5 + Momentum Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
   background: linear-gradient(180deg, #000000 0%, #072f5f 100%);
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

/* Tabs – default text color */
.stTabs [data-baseweb="tab-list"] button {
    color: #cbeaff !important;
}

/* Tabs – active tab text + underline */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #58cced !important;
    border-bottom: 3px solid #3895d3 !important;
}

/* Tabs – override default highlight bar */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #3895d3 !important;
}

/* Small gap between tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

st.title("FF5 + Momentum Regression")
st.write("Upload a portfolio CSV, pull FF5 + momentum factors from WRDS, and run a regression.")


def _parse_ken_french_table(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(","):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row in Ken French file.")
    table_lines = [lines[header_idx]]
    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if not s:
            break
        first = s.split(",")[0].strip()
        if not (first.isdigit() and len(first) == 6):
            break
        table_lines.append(ln)
    csv_block = "\n".join(table_lines)
    df = pd.read_csv(io.StringIO(csv_block))
    first_col = df.columns[0]
    if first_col != "Date":
        df = df.rename(columns={first_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([-99.99, -999], np.nan) / 100.0
    return df


wrds_user = st.text_input("WRDS Username")
wrds_pass = st.text_input("WRDS Password", type="password")

if not wrds_user or not wrds_pass:
    st.info("Enter your WRDS username and password to connect.")
    st.stop()
else:

    @st.cache_data
    def load_ff6_daily(start_date: str, end_date: str, username: str, password: str) -> pd.DataFrame:
        conn = wrds.Connection(wrds_username=username, password=password)
        try:
            query = f"""
                SELECT
                    a.date,
                    a.mktrf,
                    a.smb,
                    a.hml,
                    a.rmw,
                    a.cma,
                    a.rf,
                    b.umd
                FROM ff_all.fivefactors_daily a
                LEFT JOIN ff_all.factors_daily b
                    ON a.date = b.date
                WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY a.date;
            """
            ff6 = conn.raw_sql(query, date_cols=["date"])
        finally:
            conn.close()

        num_cols = ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
        for col in num_cols:
            ff6[col] = pd.to_numeric(ff6[col], errors="coerce")
        ff6[num_cols] = ff6[num_cols].replace([-99.99, -999], np.nan)

        mean_abs_mkt = ff6["mktrf"].abs().mean(skipna=True)
        if pd.notna(mean_abs_mkt) and mean_abs_mkt > 0.05:
            ff6[num_cols] = ff6[num_cols] / 100.0

        return ff6

    @st.cache_data
    def load_ff6_monthly(start_date: str, end_date: str, username: str, password: str) -> pd.DataFrame:
        conn = wrds.Connection(wrds_username=username, password=password)
        try:
            query = f"""
                SELECT
                    a.date,
                    a.mktrf,
                    a.smb,
                    a.hml,
                    a.rmw,
                    a.cma,
                    a.rf,
                    b.umd
                FROM ff_all.fivefactors_monthly a
                LEFT JOIN ff_all.factors_monthly b
                    ON a.date = b.date
                WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY a.date;
            """
            ff6m = conn.raw_sql(query, date_cols=["date"])
        finally:
            conn.close()

        num_cols = ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
        for col in num_cols:
            ff6m[col] = pd.to_numeric(ff6m[col], errors="coerce")
        ff6m[num_cols] = ff6m[num_cols].replace([-99.99, -999], np.nan)

        mean_abs_mkt = ff6m["mktrf"].abs().mean(skipna=True)
        if pd.notna(mean_abs_mkt) and mean_abs_mkt > 0.05:
            ff6m[num_cols] = ff6m[num_cols] / 100.0

        return ff6m

    def _load_permno_mapping_from_conn(conn, tickers, end_date: str) -> pd.DataFrame:
        tickers_sql = ",".join([f"'{t}'" for t in tickers])
        last_names = pd.DataFrame()
        tried_cols = []

        for candidate_col in ["tic", "ticker", "tkr", "symbol"]:
            try:
                tried_cols.append(candidate_col)
                q = f"""
                    SELECT permno, {candidate_col} AS ticker, namedt
                    FROM crsp.msenames
                    WHERE {candidate_col} IN ({tickers_sql})
                      AND namedt <= '{end_date}'
                """
                names = conn.raw_sql(q, date_cols=["namedt"])
                if not names.empty:
                    last_names = names
                    break
            except Exception:
                continue

        if last_names.empty:
            raise ValueError(
                "Could not map tickers to CRSP permnos. "
                f"Tried columns: {', '.join(tried_cols)}. "
                "Check CRSP access and ticker names."
            )

        last_names["ticker"] = last_names["ticker"].str.strip()
        last_names = last_names.dropna(subset=["permno", "ticker"])
        last_names = last_names[last_names["ticker"].isin(tickers)]

        if last_names.empty:
            raise ValueError("No permno mapping found for the given tickers.")

        last_names = (
            last_names.sort_values(["ticker", "namedt"])
            .groupby("ticker")
            .tail(1)
        )
        return last_names[["ticker", "permno"]]

    def build_portfolio_from_weights(raw_df: pd.DataFrame, cols_lower: dict, start_date, end_date, username: str, password: str) -> pd.DataFrame:
        conn = wrds.Connection(wrds_username=username, password=password)
        try:
            ticker_col = cols_lower.get("ticker")
            weight_col = cols_lower.get("weight")
            if ticker_col is None or weight_col is None:
                raise ValueError("Weights file must contain 'Ticker' and 'Weight' columns (any casing).")

            weights = raw_df[[ticker_col, weight_col]].copy()
            weights = weights.rename(columns={ticker_col: "Ticker", weight_col: "Weight"})
            weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")
            weights = weights.dropna(subset=["Ticker", "Weight"])
            if weights.empty:
                raise ValueError("No valid weights after cleaning.")
            weights["Weight"] = weights["Weight"] / weights["Weight"].sum()

            tickers = weights["Ticker"].unique().tolist()
            mapping = _load_permno_mapping_from_conn(conn, tickers, end_date.strftime("%Y-%m-%d"))

            weights_m = weights.merge(mapping, left_on="Ticker", right_on="ticker", how="inner")
            if weights_m.empty:
                raise ValueError("No overlap between weights tickers and CRSP msenames tickers.")

            permnos = weights_m["permno"].unique().tolist()
            permno_sql = ",".join(str(int(p)) for p in permnos)
            crsp_q = f"""
                SELECT date, permno, ret
                FROM crsp.dsf
                WHERE date BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
                  AND permno IN ({permno_sql})
                ORDER BY date;
            """
            crsp = conn.raw_sql(crsp_q, date_cols=["date"])
            crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
            crsp = crsp.dropna(subset=["ret"])
            if crsp.empty:
                raise ValueError("No valid CRSP returns for the given permnos and date range.")

            crsp = crsp.merge(
                weights_m[["permno", "Weight"]],
                on="permno",
                how="inner"
            )
            if crsp.empty:
                raise ValueError("No overlap between CRSP data and weights after merge.")

            crsp["weighted_ret"] = crsp["ret"] * crsp["Weight"]
            port = (
                crsp.groupby("date", as_index=False)["weighted_ret"]
                .sum()
                .rename(columns={"weighted_ret": "portfolio_ret"})
            )
            port = port.sort_values("date")
            return port
        finally:
            conn.close()

    uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])
    st.caption("CSV must have a `date` column and either a return column or a price column. Ken French CSVs are also supported.")

    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            first_line = raw_bytes.splitlines()[0].decode("latin1").strip()

            if first_line.startswith("This file was created using"):
                text = raw_bytes.decode("latin1")
                df_all = _parse_ken_french_table(text)
                ret_cols = [c for c in df_all.columns if c != "Date"]
                if not ret_cols:
                    st.error("No return columns found in Ken French file.")
                    st.stop()
                chosen = st.selectbox("Choose Ken French portfolio series", ret_cols)
                port = (
                    df_all[["Date", chosen]]
                    .rename(columns={"Date": "date", chosen: "portfolio_ret"})
                    .dropna()
                )
                st.write(f"Detected Ken French file. Using series: {chosen}")
            else:
                port = pd.read_csv(io.BytesIO(raw_bytes), on_bad_lines="skip")
                cols_lower = {c.lower(): c for c in port.columns}

                if "ticker" in cols_lower and "weight" in cols_lower and "date" not in cols_lower:
                    st.info("Detected holdings file (Ticker, Weight). Building daily portfolio returns from CRSP.")
                    default_start = pd.to_datetime("2010-01-01").date()
                    default_end = pd.to_datetime("2025-12-31").date()
                    start_hold = st.date_input("Holdings start date", value=default_start)
                    end_hold = st.date_input("Holdings end date", value=default_end)
                    if start_hold >= end_hold:
                        st.error("Start date must be before end date.")
                        st.stop()
                    port = build_portfolio_from_weights(port, cols_lower, start_hold, end_hold, wrds_user, wrds_pass)
                    cols_lower = {c.lower(): c for c in port.columns}

                if "date" in cols_lower:
                    date_col = cols_lower["date"]
                else:
                    st.error("No 'date' column found.")
                    st.stop()
                port[date_col] = pd.to_datetime(port[date_col])
                port = port.sort_values(date_col)

                return_candidates = ["return", "ret", "portfolio_return", "portfolio_ret",
                                     "excess_ret", "excess_return"]
                return_col = None
                for name in return_candidates:
                    if name in cols_lower:
                        return_col = cols_lower[name]
                        break

                if return_col is not None:
                    port["portfolio_ret"] = pd.to_numeric(port[return_col], errors="coerce")
                else:
                    price_candidates = ["price", "close", "adj_close"]
                    price_col = None
                    for name in price_candidates:
                        if name in cols_lower:
                            price_col = cols_lower[name]
                            break
                    if price_col is None:
                        st.error("No return or price column found.")
                        st.stop()
                    port[price_col] = pd.to_numeric(port[price_col], errors="coerce")
                    port["portfolio_ret"] = port[price_col].pct_change()

                port = port.dropna(subset=["portfolio_ret"]
                                   )
                port = port[[date_col, "portfolio_ret"]].rename(columns={date_col: "date"})

            freq_choice = st.radio(
                "Regression frequency",
                ["Monthly (FF5 + UMD)", "Daily (FF5 + UMD)"]
            )

            if freq_choice.startswith("Daily"):
                start_date = port["date"].min().strftime("%Y-%m-%d")
                end_date = port["date"].max().strftime("%Y-%m-%d")
                ff6 = load_ff6_daily(start_date, end_date, wrds_user, wrds_pass)
                df = port.merge(ff6, on="date", how="inner")
                periods_per_year = 252
                freq_label = "Daily"
            else:
                port_m = (
                    port
                    .set_index("date")["portfolio_ret"]
                    .resample("M")
                    .apply(lambda s: (1 + s).prod() - 1)
                    .dropna()
                    .reset_index()
                )
                start_date = port_m["date"].min().strftime("%Y-%m-%d")
                end_date = port_m["date"].max().strftime("%Y-%m-%d")
                ff6 = load_ff6_monthly(start_date, end_date, wrds_user, wrds_pass)
                port_m["ym"] = port_m["date"].dt.to_period("M")
                ff6["ym"] = ff6["date"].dt.to_period("M")
                df = port_m.merge(ff6, on="ym", how="inner", suffixes=("_port", "_ff6"))
                df = df.rename(columns={"date_port": "date", "portfolio_ret_port": "portfolio_ret"})
                df = df.drop(columns=["date_ff6", "ym"])
                periods_per_year = 12
                freq_label = "Monthly"

            num_cols = ["portfolio_ret", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=num_cols)

            if df.empty:
                st.error("No overlapping observations with FF factors.")
                st.stop()

            df["excess_ret"] = df["portfolio_ret"] - df["rf"]
            X = df[["mktrf", "smb", "hml", "rmw", "cma", "umd"]].to_numpy(dtype=float)
            X = sm.add_constant(X)
            y = df["excess_ret"].to_numpy(dtype=float)

            model = sm.OLS(y, X).fit()

            # --- DLM prep (for tabs) ---
            factor_cols_dlm = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
            F_dlm = df[factor_cols_dlm].to_numpy(dtype=float)
            y_dlm = df["excess_ret"].to_numpy(dtype=float)
            res_dlm = em_dlm_multifactor(y_dlm, F_dlm)
            alpha_dlm = float(res_dlm["alpha"])
            betas_dlm_smooth = np.asarray(res_dlm["beta_smooth"], dtype=float)
            avg_betas_dlm = betas_dlm_smooth.mean(axis=0)
            avg_beta_series = pd.Series(avg_betas_dlm, index=factor_cols_dlm)

            # --- Rolling prep variables (used in Rolling tab) ---
            nobs = len(df)
            if freq_choice.startswith("Daily"):
                default_window = 252
            else:
                default_window = 36

            # --------- TABS ---------
            tab_model, tab_dlm, tab_roll = st.tabs(
                ["Model & Fit", "DLM Alpha & Betas", "Rolling Betas"]
            )

            # ----- TAB 1: Model & Fit -----
            with tab_model:
                st.subheader("Model Fit")
                fit_df = pd.DataFrame(
                    {
                        "Value": [
                            model.rsquared,
                            model.rsquared_adj,
                            model.fvalue,
                            model.f_pvalue,
                        ]
                    },
                    index=["R-squared", "Adj. R-squared", "F-statistic", "Prob(F-statistic)"],
                )
                st.write(fit_df.round(6))

                st.subheader("Coefficients")
                param_names = [
                    "Alpha",
                    "Beta_Mkt",
                    "Beta_SMB",
                    "Beta_HML",
                    "Beta_RMW",
                    "Beta_CMA",
                    "Beta_UMD",
                ]
                params = pd.Series(model.params, index=param_names)
                coeff_table = pd.DataFrame({"Coefficient": params})
                st.write(coeff_table.round(6))

                st.subheader("Factor Betas")
                betas_only = params[
                    ["Beta_Mkt", "Beta_SMB", "Beta_HML", "Beta_RMW", "Beta_CMA", "Beta_UMD"]
                ]
                betas_df = betas_only.to_frame(name="Beta")
                st.bar_chart(betas_df)

            # ----- TAB 2: DLM Alpha & Betas -----
            with tab_dlm:
                st.subheader("Dynamic Multi-Factor (DLM) Alpha")
                st.metric(f"Alpha (DLM, {freq_label.lower()})", f"{alpha_dlm:.6f}")
                st.write("Average DLM betas (smoothed):")
                st.write(avg_beta_series.round(4))

            # ----- TAB 3: Rolling Betas -----
            with tab_roll:
                st.subheader("Rolling OLS Betas and Kalman-smoothed OLS Betas")

                if nobs < 20:
                    st.warning("Not enough observations for rolling OLS betas.")
                else:
                    window = min(default_window, nobs)
                    factor_cols_roll = factor_cols_dlm
                    roll_betas = np.full((nobs, len(factor_cols_roll)), np.nan)
                    for i in range(window, nobs + 1):
                        y_win = df["excess_ret"].iloc[i - window:i].to_numpy(dtype=float)
                        X_win = df[factor_cols_roll].iloc[i - window:i].to_numpy(dtype=float)
                        X_win = sm.add_constant(X_win)
                        res_win = sm.OLS(y_win, X_win).fit()
                        roll_betas[i - 1, :] = res_win.params[1:]
                    for j, name in enumerate(factor_cols_roll):
                        df[f"{name}_beta_roll"] = roll_betas[:, j]

                    def kalman_smooth_1d(obs, q=1e-5, r=None):
                        obs = np.asarray(obs, dtype=float)
                        n = len(obs)
                        out = np.full(n, np.nan)
                        valid = ~np.isnan(obs)
                        idx = np.where(valid)[0]
                        if idx.size == 0:
                            return out
                        if r is None:
                            r = np.nanvar(obs[valid])
                            if not np.isfinite(r) or r <= 0:
                                r = 1e-4
                        x = obs[idx[0]]
                        P = 1.0
                        out[idx[0]] = x
                        Q = q
                        R = r
                        for t in range(idx[0] + 1, n):
                            x_pred = x
                            P_pred = P + Q
                            if valid[t]:
                                z = obs[t]
                                K = P_pred / (P_pred + R)
                                x = x_pred + K * (z - x_pred)
                                P = (1.0 - K) * P_pred
                            else:
                                x = x_pred
                                P = P_pred
                            out[t] = x
                        return out

                    for name in factor_cols_roll:
                        roll_series = df[f"{name}_beta_roll"].to_numpy(dtype=float)
                        df[f"{name}_beta_kalman"] = kalman_smooth_1d(roll_series)

                    plot_mode = st.radio(
                        "Rolling beta view",
                        ["By factor", "All factors"],
                        index=0,
                    )

                    df_idx = df.set_index("date")

                    if plot_mode == "By factor":
                        factor_choice = st.selectbox("Choose factor", factor_cols_roll)
                        plot_df = df_idx[
                            [f"{factor_choice}_beta_roll", f"{factor_choice}_beta_kalman"]
                        ].rename(
                            columns={
                                f"{factor_choice}_beta_roll": "Rolling OLS beta",
                                f"{factor_choice}_beta_kalman": "Kalman-smoothed beta",
                            }
                        )
                        st.line_chart(plot_df)
                    else:
                        plot_cols = []
                        for name in factor_cols_roll:
                            plot_cols.append(f"{name}_beta_roll")
                            plot_cols.append(f"{name}_beta_kalman")
                        overlap_plot = df_idx[plot_cols]
                        st.line_chart(overlap_plot)

            # --- stash for Dynamic Multi-Factor page ---
            st.session_state["dlm_result"] = res_dlm
            st.session_state["dlm_factor_cols"] = factor_cols_dlm
            st.session_state["dlm_df"] = df.copy()
            st.session_state["dlm_periods_per_year"] = periods_per_year
            st.session_state["dlm_freq_label"] = freq_label
            st.session_state["factor_cols"] = factor_cols_dlm
            st.session_state["combined_df"] = df.copy()

        except Exception as e:
            st.error(f"Error processing file or running regression: {e}")
