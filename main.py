import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

FILE_2015 = "2015SalarySurveyDATA.xlsx"
FILE_2023 = "2023SalarySurvey_DATA.xlsx"

COLUMNS_2015 = [
    "Member","EmploymentStatus","WorkFunction","Industry","JobSatisfaction",
    "LevelOfEducation","YearsOfExperience","Age","Sex","AACECertified",
    "CurrentSalaryAmount","SameEmployer","WorkHours","Travel","ProjectSize"
]

COLUMNS_2023 = [
    "Member","EmploymentStatus","WorkFunction","Industry","JobSatisfaction",
    "LevelOfEducation","YearsOfExperience","Age","Sex","AACECertified",
    "CurrentSalaryAmount","SameEmployer","WorkHours","Travel","ProjectSize"
]

EXCHANGE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.74,
    "AUD": 0.66,
    "INR": 0.012,
    "IDR": 0.000065,
    "JPY": 0.0067,
    "CNY": 0.14,
    "SGD": 0.74
}

@st.cache_data
def load_data():
    df_2015 = pd.read_excel(FILE_2015)
    df_2023 = pd.read_excel(FILE_2023)

    df_2015 = df_2015[COLUMNS_2015]
    df_2023 = df_2023[COLUMNS_2023]

    df_2015["SurveyYear"] = 2015
    df_2023["SurveyYear"] = 2023
    

    combined = pd.concat([df_2015, df_2023], ignore_index=True)

    combined["YearsOfExperience"] = pd.to_numeric(combined["YearsOfExperience"], errors="coerce")
    combined["CurrentSalaryAmount"] = pd.to_numeric(combined["CurrentSalaryAmount"], errors="coerce")
    combined["Age"] = pd.to_numeric(combined["Age"], errors="coerce")

    combined["Currency"] = combined["CurrentSalaryCurrency"].astype(str).str.upper().str.strip()

    combined["ExchangeRate"] = combined["Currency"].map(EXCHANGE_RATES_TO_USD)

    combined["SalaryUSD"] = combined["CurrentSalaryAmount"] * combined["ExchangeRate"]

    combined["IsCertified"] = combined["AACECertified"].astype(str).str.contains("Yes", case=False, na=False)
    combined["IsMember"] = combined["Member"].astype(str).str.contains("Yes", case=False, na=False)

    return combined

st.title("Salary Correlation Dashboard")

df = load_data()

st.write("Total records:", len(df))

df_plot = df.dropna(subset=["YearsOfExperience", "SalaryUSD"])
df_plot = df_plot[df_plot["SalaryUSD"] > 10000]
df_plot = df_plot[df_plot["SalaryUSD"] < 500000]

st.subheader("Salary vs Years of Experience")

fig, ax = plt.subplots(figsize=(4.8, 3))

ax.scatter(
    df_plot["YearsOfExperience"],
    df_plot["SalaryUSD"],
    alpha=0.35,
    s=14
)

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary (USD)")
ax.set_title("Salary vs Years of Experience")
ax.ticklabel_format(style="plain", axis="y")

st.pyplot(fig)

corr_exp = df_plot["YearsOfExperience"].corr(df_plot["SalaryUSD"])
st.write("Correlation coefficient:", round(corr_exp, 4))

st.subheader("Member vs Non-Member Salary Distribution")

fig2, ax2 = plt.subplots(figsize=(4.8, 3))

members = df_plot[df_plot["IsMember"] == True]
non_members = df_plot[df_plot["IsMember"] == False]

ax2.scatter(members["YearsOfExperience"], members["SalaryUSD"], alpha=0.4, s=14, label="Members")
ax2.scatter(non_members["YearsOfExperience"], non_members["SalaryUSD"], alpha=0.4, s=14, label="Non-Members")

ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Salary (USD)")
ax2.set_title("Members vs Non-Members Salary")
ax2.ticklabel_format(style="plain", axis="y")
ax2.legend()

st.pyplot(fig2)

member_salary_gap = members["SalaryUSD"].mean() - non_members["SalaryUSD"].mean()
st.write("Average salary difference (Members - Non-Members):", round(member_salary_gap, 2))

st.subheader("Certified vs Non-Certified Salary Distribution")

fig3, ax3 = plt.subplots(figsize=(4.8, 3))

certified = df_plot[df_plot["IsCertified"] == True]
not_certified = df_plot[df_plot["IsCertified"] == False]

ax3.scatter(certified["YearsOfExperience"], certified["SalaryUSD"], alpha=0.4, s=14, label="Certified")
ax3.scatter(not_certified["YearsOfExperience"], not_certified["SalaryUSD"], alpha=0.4, s=14, label="Non-Certified")

ax3.set_xlabel("Years of Experience")
ax3.set_ylabel("Salary (USD)")
ax3.set_title("Certified vs Non-Certified Salary")
ax3.ticklabel_format(style="plain", axis="y")
ax3.legend()

st.pyplot(fig3)

cert_salary_gap = certified["SalaryUSD"].mean() - not_certified["SalaryUSD"].mean()
st.write("Average salary difference (Certified - Non-Certified):", round(cert_salary_gap, 2))
