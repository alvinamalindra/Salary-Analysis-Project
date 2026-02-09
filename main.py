import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =========================
# FILE (LOCAL ONLY)
# =========================
DATA_FILE = "salary_usd_cleaned.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["SalaryUSD"] = pd.to_numeric(df["Salary_USD"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df["IsCertified"] = df["AACECertified"].astype(str).str.contains("Yes", case=False, na=False)
    df["IsMember"] = df["Member"].astype(str).str.contains("Yes", case=False, na=False)
    df["IsFemale"] = df["Sex"].astype(str).str.contains("Female", case=False, na=False)

    return df

# =========================
# APPROVED PLOT FUNCTION
# DO NOT MODIFY
# =========================
def plot_group(df, group_filter, title):
    df_group = df[group_filter].dropna(subset=["YearsOfExperience", "SalaryUSD"])
    df_group = df_group[df_group["SalaryUSD"].between(10000, 500000)]

    if len(df_group) < 10:
        st.write("Not enough data for", title)
        return

    base_mean = df_group["SalaryUSD"].mean()
    base_std = df_group["SalaryUSD"].std()

    UCL = base_mean + 3 * base_std
    LCL = base_mean - 3 * base_std

    df_plot = df_group[df_group["SalaryUSD"] <= UCL]

    x = df_plot["YearsOfExperience"]
    y = df_plot["SalaryUSD"]

    if len(x) > 5:
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        x_sorted = np.linspace(x.min(), x.max(), 200)
        y_fit = p(x_sorted)
    else:
        x_sorted = np.linspace(x.min(), x.max(), 200)
        y_fit = None

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.scatter(x, y, alpha=0.35, s=14)

    if y_fit is not None:
        ax.plot(x_sorted, y_fit, color="fuchsia", linewidth=3)

    ax.axhline(base_mean)
    ax.axhline(UCL, linestyle="--")
    ax.axhline(LCL, linestyle="--")

    # Right-side label position
    x_max = x.max()
    x_label_pos = x_max + (x_max * 0.03)

    ax.text(
        x_label_pos,
        base_mean,
        "Mean",
        va="center"
    )

    ax.text(
        x_label_pos,
        UCL,
        "UCL +3σ",
        va="center"
    )

    ax.text(
        x_label_pos,
        LCL,
        "LCL −3σ",
        va="center"
    )

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    ax.set_title(title)

    ax.ticklabel_format(style="plain", axis="y")
    plt.subplots_adjust(right=0.85)

    st.pyplot(fig)

    corr = df_plot["YearsOfExperience"].corr(df_plot["SalaryUSD"])
    r2 = corr ** 2

    st.markdown(
    f"""
    Correlation: {corr:.4f}  
    R²: {r2:.4f}  
    Mean Salary: ${base_mean:,.0f}  
    Std Deviation: ${base_std:,.0f}  
    UCL (+3σ): ${UCL:,.0f}  
    LCL (−3σ): ${LCL:,.0f}
    """
)



# =========================
# APP
# =========================
st.title("Salary Analysis Dashboard")

df = load_data()
st.write("Total records:", len(df))

df_2015 = df[df["SurveyYear"] == 2015]
df_2023 = df[df["SurveyYear"] == 2023]

# =========================
# MEMBERSHIP
# =========================
st.header("Members vs Non-Members")

c1, c2 = st.columns(2)

with c1:
    plot_group(df_2015, df_2015["IsMember"], "Members (2015)")
    plot_group(df_2015, ~df_2015["IsMember"], "Non-Members (2015)")

with c2:
    plot_group(df_2023, df_2023["IsMember"], "Members (2023)")
    plot_group(df_2023, ~df_2023["IsMember"], "Non-Members (2023)")

# =========================
# CERTIFICATION
# =========================
st.header("Certification Effect")

c1, c2 = st.columns(2)

with c1:
    plot_group(df_2015, df_2015["IsCertified"], "Certified (2015)")
    plot_group(df_2015, ~df_2015["IsCertified"], "Non-Certified (2015)")

with c2:
    plot_group(df_2023, df_2023["IsCertified"], "Certified (2023)")
    plot_group(df_2023, ~df_2023["IsCertified"], "Non-Certified (2023)")

# =========================
# GENDER
# =========================
st.header("Gender Comparison")

c1, c2 = st.columns(2)

with c1:
    plot_group(df_2015, df_2015["IsFemale"], "Women (2015)")
    plot_group(df_2015, ~df_2015["IsFemale"], "Men (2015)")

with c2:
    plot_group(df_2023, df_2023["IsFemale"], "Women (2023)")
    plot_group(df_2023, ~df_2023["IsFemale"], "Men (2023)")

# =========================
# JOB SATISFACTION
# =========================
st.header("Job Satisfaction Analysis")

df_sat = df.dropna(subset=["JobSatisfaction", "SalaryUSD"]).copy()

# Clean formatting
df_sat["JobSatisfaction"] = df_sat["JobSatisfaction"].str.strip().str.lower()

# Logical satisfaction order
order = [
    "very dissatisfied",
    "somewhat dissatisfied",
    "somewhat satisfied",
    "very satisfied"
]

# Display labels (two lines)
display_labels = [
    "Very\nDissatisfied",
    "Somewhat\nDissatisfied",
    "Somewhat\nSatisfied",
    "Very\nSatisfied"
]

# Define satisfied group
df_sat["IsSatisfied"] = df_sat["JobSatisfaction"].isin(
    ["very satisfied", "somewhat satisfied"]
)

pct_satisfied = df_sat["IsSatisfied"].mean() * 100
st.write(f"Overall Percentage Satisfied: {pct_satisfied:.1f}%")

# -------------------------
# 1. Satisfaction Distribution (2015 vs 2023)
# -------------------------
st.subheader("Job Satisfaction Distribution (2015 vs 2023)")

dist_2015 = (
    df_sat[df_sat["SurveyYear"] == 2015]["JobSatisfaction"]
    .value_counts(normalize=True) * 100
)

dist_2023 = (
    df_sat[df_sat["SurveyYear"] == 2023]["JobSatisfaction"]
    .value_counts(normalize=True) * 100
)

dist_compare = pd.concat([dist_2015, dist_2023], axis=1)
dist_compare.columns = ["2015 (%)", "2023 (%)"]
dist_compare = dist_compare.reindex(order)
dist_compare = dist_compare.fillna(0)

fig, ax = plt.subplots(figsize=(8, 4))
dist_compare.plot(kind="bar", ax=ax)

ax.set_ylabel("Percentage of Participants")
ax.set_xlabel("Satisfaction Level")
ax.set_title("Job Satisfaction Comparison (2015 vs 2023)")
ax.grid(axis="y", linestyle="--", alpha=0.6)

ax.set_xticklabels(display_labels, rotation=0)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Smaller percentage labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=8)

plt.tight_layout()
st.pyplot(fig)

st.dataframe(dist_compare.round(2))


# -------------------------
# 2. Salary by Satisfaction (2015 vs 2023)
# -------------------------
st.subheader("Average Salary by Satisfaction Level")

salary_2015 = (
    df_sat[df_sat["SurveyYear"] == 2015]
    .groupby("JobSatisfaction")["SalaryUSD"]
    .mean()
)

salary_2023 = (
    df_sat[df_sat["SurveyYear"] == 2023]
    .groupby("JobSatisfaction")["SalaryUSD"]
    .mean()
)

salary_compare = pd.concat([salary_2015, salary_2023], axis=1)
salary_compare.columns = ["2015 Avg Salary", "2023 Avg Salary"]
salary_compare = salary_compare.reindex(order)
salary_compare = salary_compare.fillna(0)

fig2, ax2 = plt.subplots(figsize=(8, 4))
salary_compare.plot(kind="bar", ax=ax2)

ax2.set_ylabel("Average Salary (USD)")
ax2.set_xlabel("Satisfaction Level")
ax2.set_title("Salary by Job Satisfaction (2015 vs 2023)")
ax2.grid(axis="y", linestyle="--", alpha=0.6)

# Set Y-axis range for cleaner look
ax2.set_ylim(55000, 145000)

ax2.set_xticklabels(display_labels, rotation=0)
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Smaller salary labels
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

plt.tight_layout()
st.pyplot(fig2)

st.dataframe(salary_compare.round(0))

# =========================
# GENDER GAP BY EDUCATION
# =========================
st.header("Gender Salary Gap by Education Level")

df_gender = df.dropna(subset=["SalaryUSD", "LevelOfEducation"]).copy()

# -------------------------
# CLEAN & STANDARDIZE EDUCATION LEVELS
# -------------------------
df_gender["LevelOfEducation"] = (
    df_gender["LevelOfEducation"]
    .str.strip()
    .str.lower()
)

# Fix encoding issues and duplicates
df_gender["LevelOfEducation"] = df_gender["LevelOfEducation"].replace({
    "undergraduate/bachelor���s degree": "undergraduate or bachelors degree",
    "undergraduate/bachelor's degree": "undergraduate or bachelors degree",
    "undergraduate or bachelor’s degree": "undergraduate or bachelors degree",
    "graduate/master���s degree": "graduate - masters degree",
    "graduate/master's degree": "graduate - masters degree",
    "graduate/doctoral degree": "graduate - doctoral degree"
})

# -------------------------
# DEFINE LOGICAL EDUCATION ORDER
# -------------------------
edu_order = [
    "high school",
    "associate degree",
    "undergraduate or bachelors degree",
    "graduate - masters degree",
    "graduate - doctoral degree"
]

# -------------------------
# CALCULATE AVERAGES
# -------------------------
gender_edu = (
    df_gender
    .groupby(["LevelOfEducation", "IsFemale"])["SalaryUSD"]
    .mean()
    .unstack()
)

gender_edu.columns = ["Men Avg Salary", "Women Avg Salary"]

# Reorder properly
gender_edu = gender_edu.reindex(edu_order)

# Calculate % gap
gender_edu["Gap % (Women vs Men)"] = (
    (gender_edu["Women Avg Salary"] - gender_edu["Men Avg Salary"])
    / gender_edu["Men Avg Salary"]
) * 100

st.dataframe(gender_edu.round(2))

# -------------------------
# MULTI-LINE DISPLAY LABELS
# -------------------------
display_labels = [
    "High\nSchool",
    "Associate\nDegree",
    "Undergraduate\nBachelor's",
    "Graduate\nMaster's",
    "Graduate\nDoctoral"
]

# -------------------------
# BAR CHART
# -------------------------
fig, ax = plt.subplots(figsize=(10, 4))

gender_edu[["Men Avg Salary", "Women Avg Salary"]].plot(
    kind="bar",
    ax=ax
)

ax.set_ylabel("Average Salary (USD)")
ax.set_xlabel("Education Level")
ax.set_title("Average Salary by Gender and Education")
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Replace x-axis labels
ax.set_xticklabels(display_labels, rotation=0)

# Legend outside
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Smaller data labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

plt.tight_layout()
st.pyplot(fig)

# =========================
# MULTIVARIATE CORRELATION
# =========================
st.header("Multivariate Correlation Model")

reg_df = df[
    [
        "SalaryUSD",
        "YearsOfExperience",
        "IsCertified",
        "IsMember",
        "IsFemale"
    ]
].dropna()

X = reg_df[
    [
        "YearsOfExperience",
        "IsCertified",
        "IsMember",
        "IsFemale"
    ]
].astype(float)

X = sm.add_constant(X)
y = reg_df["SalaryUSD"]

model = sm.OLS(y, X).fit()
st.text(model.summary())

# =========================
# DEBUG: SHOW ALL COLUMNS
# =========================
st.header("Debug: Column Names")

st.write("All Columns in Dataset:")
st.write(df.columns.tolist())
