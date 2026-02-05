import streamlit as st
import pandas as pd

# =========================
# FILE PATHS
# =========================
SALARY_2015 = "2015SalarySurveyDATA.xlsx"
SALARY_2023 = "2023SalarySurvey_DATA.xlsx"
ISO_FILE = "currency_ISO.xlsx"
FX_FILE = "fx_rates_merged_2015_2023.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_salary():
    df15 = pd.read_excel(SALARY_2015)
    df23 = pd.read_excel(SALARY_2023)

    df15["SurveyYear"] = 2015
    df23["SurveyYear"] = 2023

    return pd.concat([df15, df23], ignore_index=True)


@st.cache_data
def load_iso():
    return pd.read_excel(ISO_FILE)


@st.cache_data
def load_fx():
    fx = pd.read_csv(FX_FILE)

    fx_long = fx.melt(
        id_vars="Currency",
        var_name="SurveyYear",
        value_name="FX_rate"
    )

    fx_long["SurveyYear"] = fx_long["SurveyYear"].astype(int)
    fx_long = fx_long.rename(columns={"Currency": "CurrencyISO"})

    return fx_long


# =========================
# APP
# =========================
st.title("Salary Currency Check & USD Conversion")

salary = load_salary()
iso = load_iso()
fx = load_fx()

# =========================
# STEP 1: MAP CURRENCY TO ISO
# =========================
salary_iso = salary.merge(
    iso,
    left_on="CurrentSalaryCurrency",
    right_on="CountryCurrency",
    how="left"
)

salary_iso = salary_iso.rename(columns={"ISO": "CurrencyISO"})

st.subheader("Salary Data with ISO Currency")
st.dataframe(
    salary_iso[
        ["SurveyYear", "CurrentSalaryCurrency", "CurrencyISO"]
    ]
)

# =========================
# STEP 2: UNMAPPED CURRENCIES
# =========================
st.subheader("Unmapped Currency Values")

unmapped = (
    salary_iso[salary_iso["CurrencyISO"].isna()]
    .groupby(["SurveyYear", "CurrentSalaryCurrency"])
    .size()
    .reset_index(name="Count")
)

if unmapped.empty:
    st.success("All salary currencies are mapped to ISO codes.")
else:
    st.error("Unmapped currency values found.")
    st.dataframe(unmapped)

# =========================
# STEP 3: FX COVERAGE CHECK
# =========================
st.subheader("FX Coverage Check")

salary_fx = salary_iso.merge(
    fx,
    on=["CurrencyISO", "SurveyYear"],
    how="left"
)

missing_fx = (
    salary_fx[salary_fx["FX_rate"].isna()]
    .groupby(["SurveyYear", "CurrencyISO"])
    .size()
    .reset_index(name="Count")
)

if missing_fx.empty:
    st.success("All currencies have FX rates for their survey year.")
else:
    st.error("Missing FX rates detected.")
    st.dataframe(missing_fx)

# =========================
# STEP 4: CONVERT TO USD
# =========================
salary_fx["Salary_USD"] = (
    salary_fx["CurrentSalaryAmount"] / salary_fx["FX_rate"]
)

st.subheader("Sample: Salary Converted to USD")
st.dataframe(
    salary_fx[
        [
            "SurveyYear",
            "CurrentSalaryAmount",
            "CurrentSalaryCurrency",
            "FX_rate",
            "Salary_USD",
        ]
    ].head(20)
)

# =========================
# EXPORT FILES
# =========================
st.subheader("Export Results")

if st.button("Export Check Files"):
    unmapped.to_csv("unmapped_currency.csv", index=False)
    missing_fx.to_csv("missing_fx.csv", index=False)
    st.success("Check files exported.")

if st.button("Export Final USD Salary Data"):
    salary_fx.to_csv("salary_usd_cleaned.csv", index=False)
    st.success("Final salary file exported.")
