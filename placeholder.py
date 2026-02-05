import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

df_2015 = pd.read_excel("2015SalarySurveyDATA.xlsx")
df_2023 = pd.read_excel("2023SalarySurvey_DATA.xlsx")

st.title("RAW COLUMN INSPECTION")

st.subheader("2015 columns")
st.write(list(df_2015.columns))

st.subheader("2023 columns")
st.write(list(df_2023.columns))