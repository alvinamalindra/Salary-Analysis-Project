import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

st.set_page_config(page_title="Salary Survey Research", layout="wide")
st.title("📊 AACE Salary Survey — Correlation & Causation Research")

@st.cache_data
def load_data():
    df_2015 = pd.read_excel("2015SalarySurveyDATA.xlsx")
    df_2023 = pd.read_excel("2023SalarySurvey_DATA.xlsx")
    df_2015["Year"] = 2015
    df_2023["Year"] = 2023
    return df_2015, df_2023

df_2015, df_2023 = load_data()

# --- Standardize columns ---
rename_map = {
    "YearsOfExperience": "Experience",
    "LevelOfEducation": "Education",
    "Sex": "Gender"
}

df_2015 = df_2015.rename(columns=rename_map)
df_2023 = df_2023.rename(columns=rename_map)

# --- Certification Flag ---
cert_cols = [c for c in df_2023.columns if c.startswith("CertType")] + ["AACECertified"]

df_2015["IsCertified"] = (df_2015["AACECertified"] == "Yes")
df_2023["IsCertified"] = df_2023[cert_cols].notnull().any(axis=1)

# --- Conference / RP / Presenter Flags ---
conf_col = "Have you attended an AACE Conference & Expo (Annual Meeting)?"
paper_col = "Have you presented a paper at an AACE Conference & Expo (Annual Meeting)?"
rp_col = "Have you contributed to an AACE recommended practice?"

df_2023["AttendedConference"] = df_2023[conf_col] == "Yes"
df_2023["PresentedPaper"] = df_2023[paper_col] == "Yes"
df_2023["ContributedRP"] = df_2023[rp_col] == "Yes"

df_2015["AttendedConference"] = np.nan
df_2015["PresentedPaper"] = np.nan
df_2015["ContributedRP"] = np.nan

# --- Combine ---
combined = pd.concat([df_2015, df_2023], ignore_index=True)

combined["CurrentSalaryAmount"] = pd.to_numeric(combined["CurrentSalaryAmount"], errors="coerce")

st.subheader("📁 Combined Dataset Preview")
st.dataframe(combined[[
    "Year","CurrentSalaryAmount","Member","IsCertified","Experience",
    "Education","Gender","AttendedConference","PresentedPaper","ContributedRP"
]].head())

# =====================
# CORRELATION RANKING
# =====================
st.subheader("📈 Correlation Ranking (Salary Drivers)")

corr_factors = [
    "Member","IsCertified","Experience","AttendedConference",
    "PresentedPaper","ContributedRP"
]

corr_results = []

for col in corr_factors:
    temp = combined[[col,"CurrentSalaryAmount"]].dropna()
    if len(temp) > 10:
        temp[col] = pd.factorize(temp[col])[0]
        r, p = pearsonr(temp[col], temp["CurrentSalaryAmount"])
        corr_results.append((col, r, p))

corr_df = pd.DataFrame(corr_results, columns=["Factor","Correlation","P-Value"])
corr_df["Strength"] = corr_df["Correlation"].abs()
corr_df = corr_df.sort_values("Strength", ascending=False)

st.dataframe(corr_df)

# =====================
# CAUSATION REGRESSION
# =====================
st.subheader("Causation Model (Multiple Regression)")

reg_cols = corr_factors + ["Education"]

reg_df = combined[["CurrentSalaryAmount"] + reg_cols].dropna()

for c in reg_cols:
    reg_df[c] = pd.factorize(reg_df[c])[0]

X = sm.add_constant(reg_df[reg_cols])
y = reg_df["CurrentSalaryAmount"]

model = sm.OLS(y, X).fit()

st.text(model.summary())

# =====================
# TOP CAUSATION DRIVERS
# =====================
st.subheader("Top Salary Drivers (Causation Rank)")

rank_df = pd.DataFrame({
    "Factor": reg_cols,
    "Beta": model.params[1:],
    "P-Value": model.pvalues[1:]
})

rank_df["Impact"] = rank_df["Beta"].abs()
rank_df = rank_df.sort_values("Impact", ascending=False)

st.dataframe(rank_df)
