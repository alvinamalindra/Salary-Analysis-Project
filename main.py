import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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

    if len(df_group) < 10:
        st.write("Not enough data for", title)
        return

    # ---- BASELINE STATS (DO NOT CHANGE) ----
    base_mean = df_group["SalaryUSD"].mean()
    base_std = df_group["SalaryUSD"].std()

    UCL = base_mean + 3 * base_std
    LCL = base_mean - 3 * base_std

    # ---- FILTER ONLY FOR DISPLAY ----
    df_plot = df_group[df_group["SalaryUSD"] <= UCL]

    x = df_plot["YearsOfExperience"]
    y = df_plot["SalaryUSD"]

    # Best-fit polynomial
    if len(x) > 5:
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        x_sorted = np.linspace(x.min(), x.max(), 200)
        y_fit = p(x_sorted)
    else:
        x_sorted = np.linspace(x.min(), x.max(), 200)
        y_fit = None

    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    # Scatter points
    ax.scatter(x, y, alpha=0.35, s=14)

    # Best-fit curve
    if y_fit is not None:
        ax.plot(x_sorted, y_fit)

    # ---- CONTROL LINES (BASELINE, NEVER MOVE) ----
    ax.axhline(base_mean)
    ax.axhline(UCL, linestyle="--")
    ax.axhline(LCL, linestyle="--")

    # ---- Y LIMITS MUST RESPECT TRUE UCL/LCL ----
    ymin = min(y.min(), LCL) * 0.95
    ymax = max(y.max(), UCL) * 1.05
    ax.set_ylim(ymin, ymax)

    # ---- LABELS ON RIGHT ----
    x_label_pos = x_sorted.max() + (x_sorted.max() * 0.03)

    ax.text(x_label_pos, base_mean, "Mean", va="center")
    ax.text(x_label_pos, UCL, "UCL +3σ", va="center")
    ax.text(x_label_pos, LCL, "LCL −3σ", va="center")

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    ax.set_title(title)

    # Granularity 50k
    ax.set_yticks(np.arange(0, 600000, 50000))
    ax.ticklabel_format(style="plain", axis="y")

    # Remove legend box
    ax.legend([], [], frameon=False)

    plt.subplots_adjust(right=0.85)

    st.pyplot(fig)

    # ---- METRICS ----
    corr = df_plot["YearsOfExperience"].corr(df_plot["SalaryUSD"])
    r2 = corr ** 2

    st.write("Correlation:", round(corr, 4))
    st.write("R² Best-Fit:", round(r2, 4))
    st.write("Baseline Mean Salary:", round(base_mean, 2))
    st.write("Baseline UCL (+3σ):", round(UCL, 2))

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
