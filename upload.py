import streamlit as st
import pandas as pd

st.title("CSV Uploader")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_fwf(uploaded_file)

        st.success("File loaded successfully!")
        st.write(df)

        if st.checkbox("Show DataFrame info"):
            buffer = []
            df.info(buf=buffer)
            s = "\n".join(buffer)
            st.text(s)

    except Exception as e:
        st.error(f"Error loading file: {e}")
