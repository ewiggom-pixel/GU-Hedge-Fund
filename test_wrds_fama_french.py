import streamlit as st
import wrds
import pandas as pd

@st.cache_data
def load_ff():
    conn = wrds.Connection()
    st.write("Connected to WRDS.")

    # Daily 5-factor
    daily = conn.raw_sql(
        """
        SELECT *
        FROM ff_all.fivefactors_daily
        ORDER BY date
        LIMIT 10;
        """,
        date_cols=["date"]
    )

    # Monthly 5-factor
    monthly = conn.raw_sql(
        """
        SELECT *
        FROM ff_all.fivefactors_monthly
        ORDER BY date
        LIMIT 10;
        """,
        date_cols=["date"]
    )

    return daily, monthly


st.title("WRDS Daily + Monthly Fama-French 5-Factor Test")

try:
    daily, monthly = load_ff()
    st.subheader("Daily (first 10 rows)")
    st.write(daily)

    st.subheader("Monthly (first 10 rows)")
    st.write(monthly)

except Exception as e:
    st.error(f"Error: {e}")
