import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

FILE_2015 = "2015SalarySurveyDATA.xlsx"
FILE_2023 = "2023SalarySurvey_DATA.xlsx"

COLUMNS = [
    "Member","EmploymentStatus","WorkFunction","Industry","JobSatisfaction",
    "LevelOfEducation","YearsOfExperience","Age","Sex","AACECertified",
    "CurrentSalaryAmount","SameEmployer","WorkHours","Travel","ProjectSize"
]

@st.cache_data
def load_data():
    df_2015 = pd.read_excel(FILE_2015)
    df_2023 = pd.read_excel(FILE_2023)

    df_2015 = df_2015[COLUMNS]
    df_2023 = df_2023[COLUMNS]

    df_2015["SurveyYear"] = 2015
    df_2023["SurveyYear"] = 2023

    combined = pd.concat([df_2015, df_2023], ignore_index=True)

    combined["YearsOfExperience"] = pd.to_numeric(combined["YearsOfExperience"], errors="coerce")
    combined["CurrentSalaryAmount"] = pd.to_numeric(combined["CurrentSalaryAmount"], errors="coerce")
    combined["Age"] = pd.to_numeric(combined["Age"], errors="coerce")

    combined["IsCertified"] = combined["AACECertified"].astype(str).str.contains("Yes", case=False, na=False)
    combined["IsMember"] = combined["Member"].astype(str).str.contains("Yes", case=False, na=False)
    combined["IsFemale"] = combined["Sex"].astype(str).str.contains("Female", case=False, na=False)

    combined["SalaryUSD"] = combined["CurrentSalaryAmount"]

    return combined


def plot_group(df, group_filter, title):
    df_group = df[group_filter].dropna(subset=["YearsOfExperience", "SalaryUSD"])

    df_group = df_group[df_group["SalaryUSD"] > 10000]
    df_group = df_group[df_group["SalaryUSD"] < 500000]

    if len(df_group) < 15:
        st.write("Not enough data for", title)
        return

    x = df_group["YearsOfExperience"]
    y = df_group["SalaryUSD"]

    mean_salary = y.mean()
    std_salary = y.std()

    coeffs = np.polyfit(x, y, 3)
    poly = np.poly1d(coeffs)

    x_sorted = np.linspace(x.min(), x.max(), 250)
    y_fit = poly(x_sorted)

    r2 = r2_score(y, poly(x))

    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    ax.scatter(x, y, alpha=0.18, s=12)

    ax.plot(x_sorted, y_fit, linewidth=3)

    ax.axhline(mean_salary, linewidth=1.3)
    ax.axhline(mean_salary + 3*std_salary, linestyle="--", linewidth=1)
    ax.axhline(mean_salary - 3*std_salary, linestyle="--", linewidth=1)

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    ax.set_title(title)
    ax.ticklabel_format(style="plain", axis="y")

    st.pyplot(fig)

    st.write("Correlation:", round(x.corr(y), 4))
    st.write("R² Best-Fit:", round(r2, 4))
    st.write("Mean Salary:", round(mean_salary, 2))


st.title("Salary Pattern Analysis Dashboard")

df = load_data()

st.write("Total records:", len(df))

st.subheader("Members Only")
plot_group(df, df["IsMember"] == True, "Members Salary vs Experience")

st.subheader("Non-Members Only")
plot_group(df, df["IsMember"] == False, "Non-Members Salary vs Experience")

st.subheader("Certified Only")
plot_group(df, df["IsCertified"] == True, "Certified Salary vs Experience")

st.subheader("Non-Certified Only")
plot_group(df, df["IsCertified"] == False, "Non-Certified Salary vs Experience")

st.subheader("Women Only")
plot_group(df, df["IsFemale"] == True, "Women Salary vs Experience")

st.subheader("Men Only")
plot_group(df, df["IsFemale"] == False, "Men Salary vs Experience")
