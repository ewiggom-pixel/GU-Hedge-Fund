import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import wrds

st.set_page_config(
    page_title="FF5 + Momentum Regression",
    layout="centered"
)

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

@st.cache_resource
def get_wrds_conn():
    return wrds.Connection()

@st.cache_data
def load_ff6_daily(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_wrds_conn()
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
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        ff6[col] = pd.to_numeric(ff6[col], errors="coerce")
    return ff6

@st.cache_data
def load_ff6_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_wrds_conn()
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
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        ff6m[col] = pd.to_numeric(ff6m[col], errors="coerce")
    return ff6m

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
            chosen = ret_cols[0]
            port = (
                df_all[["Date", chosen]]
                .rename(columns={"Date": "date", chosen: "portfolio_ret"})
                .dropna()
            )
            st.write(f"Detected Ken French file. Using series: {chosen}")
        else:
            port = pd.read_csv(io.BytesIO(raw_bytes), on_bad_lines="skip")
            cols_lower = {c.lower(): c for c in port.columns}
            if "date" in cols_lower:
                date_col = cols_lower["date"]
            else:
                st.error("No 'date' column found.")
                st.stop()
            port[date_col] = pd.to_datetime(port[date_col])
            port = port.sort_values(date_col)

            return_candidates = ["return", "ret", "portfolio_return", "portfolio_ret", "excess_ret"]
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

            port = port.dropna(subset=["portfolio_ret"])
            port = port[[date_col, "portfolio_ret"]].rename(columns={date_col: "date"})

        st.write("Processed portfolio (first 5 rows):")
        st.write(port.head())

        freq_choice = st.radio(
            "Regression frequency",
            ["Daily (FF5 + UMD)", "Monthly (FF5 + UMD)"]
        )

        if freq_choice.startswith("Daily"):
            start_date = port["date"].min().strftime("%Y-%m-%d")
            end_date = port["date"].max().strftime("%Y-%m-%d")
            ff6 = load_ff6_daily(start_date, end_date)
            df = port.merge(ff6, on="date", how="inner")
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
            ff6 = load_ff6_monthly(start_date, end_date)
            df = port_m.merge(ff6, on="date", how="inner")

        num_cols = ["portfolio_ret", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=num_cols)

        if df.empty:
            st.error("No overlapping observations with FF factors.")
            st.stop()

        st.write(f"Overlapping observations: {len(df)}")

        df["excess_ret"] = df["portfolio_ret"] - df["rf"]
        X = df[["mktrf", "smb", "hml", "rmw", "cma", "umd"]].to_numpy(dtype=float)
        X = sm.add_constant(X)
        y = df["excess_ret"].to_numpy(dtype=float)

        model = sm.OLS(y, X).fit()

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

        st.subheader("Raw statsmodels summary")
        st.text(model.summary().as_text())

    except Exception as e:
        st.error(f"Error processing file or running regression: {e}")
