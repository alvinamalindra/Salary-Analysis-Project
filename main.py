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

    # -------------------------
    # CLEAN EMPLOYMENT STATUS
    # -------------------------
    df["EmploymentStatus"] = df["EmploymentStatus"].astype(str).str.strip().str.lower()

    df["EmploymentStatus"] = df["EmploymentStatus"].replace({
        "employed full-time": "full-time"
    })

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
    plt.close(fig)


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

# =========================
# GLOBAL FILTERS
# =========================
st.sidebar.header("Filters")

# Employment Status Filter
employment_options = sorted(df["EmploymentStatus"].dropna().unique())
selected_employment = st.sidebar.multiselect(
    "Employment Status",
    employment_options,
    default=employment_options
)

# Location Work Filter
location_options = sorted(df["LocationWork"].dropna().unique())
selected_location = st.sidebar.multiselect(
    "Work Location",
    location_options,
    default=location_options
)

# Apply Filters
df = df[
    (df["EmploymentStatus"].isin(selected_employment)) &
    (df["LocationWork"].isin(selected_location))
]

st.write("Filtered records:", len(df))

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
plt.close(fig)

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
st.pyplot(fig)
plt.close(fig)

st.dataframe(salary_compare.round(0))

# -------------------------
# 3. Salary by Satisfaction and Gender (2015 vs 2023)
# -------------------------
st.subheader("Salary by Satisfaction and Gender (2015 vs 2023)")

c1, c2 = st.columns(2)

def plot_sat_gender(year, container):

    df_year = df_sat[df_sat["SurveyYear"] == year]

    gender_sat = (
        df_year
        .groupby(["JobSatisfaction", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_sat.columns = ["Men Avg Salary", "Women Avg Salary"]
    gender_sat = gender_sat.reindex(order)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    gender_sat.plot(kind="bar", ax=ax)

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Satisfaction Level")
    ax.set_title(f"{year}")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.set_xticklabels(display_labels, rotation=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container_bar in ax.containers:
        ax.bar_label(container_bar, fmt="%.0f", padding=2, fontsize=7)

    plt.tight_layout()
    container.pyplot(fig)


with c1:
    plot_sat_gender(2015, st)

with c2:
    plot_sat_gender(2023, st)


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
plt.close(fig)

# =========================
# GENDER GAP BY MANAGERIAL ROLE
# =========================
st.header("Gender Salary Gap by Managerial Role")

df_mgr = df.dropna(subset=["SalaryUSD", "ManagerialDuties"]).copy()

gender_mgr = (
    df_mgr
    .groupby(["ManagerialDuties", "IsFemale"])["SalaryUSD"]
    .mean()
    .unstack()
)

gender_mgr.columns = ["Men Avg Salary", "Women Avg Salary"]

gender_mgr["Gap %"] = (
    (gender_mgr["Women Avg Salary"] - gender_mgr["Men Avg Salary"])
    / gender_mgr["Men Avg Salary"]
) * 100

st.dataframe(gender_mgr.round(2))

# =========================
# GENDER GAP BY EDUCATION + MANAGERIAL (2015 vs 2023)
# =========================
st.header("Gender Salary Gap by Education and Managerial Duties (2015 vs 2023)")

df_mgr = df.dropna(subset=["SalaryUSD", "LevelOfEducation", "ManagerialDuties"]).copy()

# Clean education text (same standardization you used earlier)
df_mgr["LevelOfEducation"] = (
    df_mgr["LevelOfEducation"]
    .str.strip()
    .str.lower()
)

df_mgr["LevelOfEducation"] = df_mgr["LevelOfEducation"].replace({
    "undergraduate/bachelor���s degree": "undergraduate or bachelors degree",
    "undergraduate/bachelor's degree": "undergraduate or bachelors degree",
    "undergraduate or bachelor’s degree": "undergraduate or bachelors degree",
    "graduate/master���s degree": "graduate - masters degree",
    "graduate/master's degree": "graduate - masters degree",
    "graduate/doctoral degree": "graduate - doctoral degree"
})

edu_order = [
    "high school",
    "associate degree",
    "undergraduate or bachelors degree",
    "graduate - masters degree",
    "graduate - doctoral degree"
]

display_labels = [
    "High\nSchool",
    "Associate\nDegree",
    "Undergraduate\nBachelor's",
    "Graduate\nMaster's",
    "Graduate\nDoctoral"
]

# Convert managerial to boolean
df_mgr["IsManager"] = df_mgr["ManagerialDuties"].astype(str).str.contains("Yes", case=False, na=False)

# Function to build chart per year
def plot_mgr_year(year):

    st.subheader(f"{year} – Managers Only")

    df_year = df_mgr[(df_mgr["SurveyYear"] == year) & (df_mgr["IsManager"] == True)]

    gender_edu_mgr = (
        df_year
        .groupby(["LevelOfEducation", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_edu_mgr.columns = ["Men Avg Salary", "Women Avg Salary"]

    gender_edu_mgr = gender_edu_mgr.reindex(edu_order)

    gender_edu_mgr["Gap %"] = (
        (gender_edu_mgr["Women Avg Salary"] - gender_edu_mgr["Men Avg Salary"])
        / gender_edu_mgr["Men Avg Salary"]
    ) * 100

    st.dataframe(gender_edu_mgr.round(2))

    fig, ax = plt.subplots(figsize=(10, 4))

    gender_edu_mgr[["Men Avg Salary", "Women Avg Salary"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Education Level")
    ax.set_title(f"Managers – Average Salary by Gender and Education ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.set_xticklabels(display_labels, rotation=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Plot for both years
plot_mgr_year(2015)
plot_mgr_year(2023)

# =========================
# GENDER GAP BY CERTIFICATION
# =========================
st.header("Gender Salary Gap by Certification")

gender_cert = (
    df
    .groupby(["IsCertified", "IsFemale"])["SalaryUSD"]
    .mean()
    .unstack()
)

gender_cert.columns = ["Men Avg Salary", "Women Avg Salary"]

gender_cert["Gap %"] = (
    (gender_cert["Women Avg Salary"] - gender_cert["Men Avg Salary"])
    / gender_cert["Men Avg Salary"]
) * 100

st.dataframe(gender_cert.round(2))

# =========================
# GENDER GAP BY EDUCATION + CERTIFICATION (2015 vs 2023)
# =========================
st.header("Gender Salary Gap by Education and Certification (2015 vs 2023)")

df_cert = df.dropna(subset=["SalaryUSD", "LevelOfEducation", "IsCertified"]).copy()

# Clean education text (same standardization)
df_cert["LevelOfEducation"] = (
    df_cert["LevelOfEducation"]
    .str.strip()
    .str.lower()
)

df_cert["LevelOfEducation"] = df_cert["LevelOfEducation"].replace({
    "undergraduate/bachelor���s degree": "undergraduate or bachelors degree",
    "undergraduate/bachelor's degree": "undergraduate or bachelors degree",
    "undergraduate or bachelor’s degree": "undergraduate or bachelors degree",
    "graduate/master���s degree": "graduate - masters degree",
    "graduate/master's degree": "graduate - masters degree",
    "graduate/doctoral degree": "graduate - doctoral degree"
})

edu_order = [
    "high school",
    "associate degree",
    "undergraduate or bachelors degree",
    "graduate - masters degree",
    "graduate - doctoral degree"
]

display_labels = [
    "High\nSchool",
    "Associate\nDegree",
    "Undergraduate\nBachelor's",
    "Graduate\nMaster's",
    "Graduate\nDoctoral"
]

# Function to build chart per year (Certified Only)
def plot_cert_year(year):

    st.subheader(f"{year} – Certified Professionals Only")

    df_year = df_cert[
        (df_cert["SurveyYear"] == year) &
        (df_cert["IsCertified"] == True)
    ]

    gender_edu_cert = (
        df_year
        .groupby(["LevelOfEducation", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_edu_cert.columns = ["Men Avg Salary", "Women Avg Salary"]

    gender_edu_cert = gender_edu_cert.reindex(edu_order)

    gender_edu_cert["Gap %"] = (
        (gender_edu_cert["Women Avg Salary"] - gender_edu_cert["Men Avg Salary"])
        / gender_edu_cert["Men Avg Salary"]
    ) * 100

    st.dataframe(gender_edu_cert.round(2))

    fig, ax = plt.subplots(figsize=(10, 4))

    gender_edu_cert[["Men Avg Salary", "Women Avg Salary"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Education Level")
    ax.set_title(f"Certified Professionals – Salary by Gender and Education ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.set_xticklabels(display_labels, rotation=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Plot for both years
plot_cert_year(2015)
plot_cert_year(2023)

# =========================
# GENDER GAP BY INDUSTRY (2015 vs 2023)
# =========================
st.header("Gender Salary Gap by Industry (2015 vs 2023)")

# -------------------------
# CLEAN INDUSTRY NAMES
# -------------------------
df_ind = df.dropna(subset=["SalaryUSD", "Industry"]).copy()

df_ind["Industry"] = df_ind["Industry"].str.strip().str.lower()

df_ind["Industry"] = df_ind["Industry"].replace({
    "power generation/utilities": "power generation or utilities",
    "oil/gas production": "oil or gas production",
    "mining/minerals": "mining or minerals",
    "enginer": "engineering",
    "other (please specify)": "other"
})

# -------------------------
# KEEP TOP INDUSTRIES ONLY
# -------------------------
top_industries = df_ind["Industry"].value_counts().head(8).index
df_ind = df_ind[df_ind["Industry"].isin(top_industries)]

# -------------------------
# FUNCTION TO FORMAT LABELS
# -------------------------
def format_label(ind):
    words = ind.title().split()
    if len(words) > 2:
        return " ".join(words[:2]) + "\n" + " ".join(words[2:])
    elif len(words) == 2:
        return words[0] + "\n" + words[1]
    else:
        return ind.title()

# -------------------------
# FUNCTION TO PLOT PER YEAR
# -------------------------
def plot_industry_year(year):

    st.subheader(f"{year}")

    df_year = df_ind[df_ind["SurveyYear"] == year]

    gender_ind = (
        df_year
        .groupby(["Industry", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_ind.columns = ["Men Avg Salary", "Women Avg Salary"]

    gender_ind["Gap %"] = (
        (gender_ind["Women Avg Salary"] - gender_ind["Men Avg Salary"])
        / gender_ind["Men Avg Salary"]
    ) * 100

    st.dataframe(gender_ind.round(2))

    fig, ax = plt.subplots(figsize=(11, 4))

    gender_ind[["Men Avg Salary", "Women Avg Salary"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Industry")
    ax.set_title(f"Average Salary by Gender and Industry ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Two-line formatted labels
    formatted_labels = [format_label(ind) for ind in gender_ind.index]
    ax.set_xticklabels(formatted_labels, rotation=0)

    # Legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Data labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -------------------------
# PLOT BOTH YEARS
# -------------------------
plot_industry_year(2015)
plot_industry_year(2023)

# =========================
# GENDER GAP BY CONSULTING STATUS (2015 vs 2023)
# =========================
st.header("Gender Salary Gap by Consulting Status (2015 vs 2023)")

# Clean consulting column
df_consult = df.dropna(subset=["SalaryUSD", "Consult"]).copy()

df_consult["IsConsultant"] = (
    df_consult["Consult"]
    .astype(str)
    .str.contains("Yes", case=False, na=False)
)

def plot_consult_year(year):

    st.subheader(f"{year}")

    df_year = df_consult[df_consult["SurveyYear"] == year]

    gender_consult = (
        df_year
        .groupby(["IsConsultant", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_consult.columns = ["Men Avg Salary", "Women Avg Salary"]

    gender_consult["Gap %"] = (
        (gender_consult["Women Avg Salary"] - gender_consult["Men Avg Salary"])
        / gender_consult["Men Avg Salary"]
    ) * 100

    # Rename index for display
    gender_consult.index = ["Non-Consultant", "Consultant"]

    st.dataframe(gender_consult.round(2))

    fig, ax = plt.subplots(figsize=(8, 4))

    gender_consult[["Men Avg Salary", "Women Avg Salary"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_xticklabels(gender_consult.index, rotation=0)

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Consulting Status")
    ax.set_title(f"Average Salary by Gender and Consulting Status ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


plot_consult_year(2015)
plot_consult_year(2023)


# =========================
# GENDER GAP BY PROJECT TYPE (2015 vs 2023)
# =========================
st.header("Gender Salary Gap by Project Type (2015 vs 2023)")

project_cols = [
    "ProjectType1","ProjectType2","ProjectType3","ProjectType4",
    "ProjectType5","ProjectType6","ProjectType7","ProjectType8",
    "ProjectType9","ProjectType10","ProjectType11","ProjectType12",
    "ProjectType13","ProjectType14"
]

df_proj = df.dropna(subset=["SalaryUSD"]).copy()

df_proj_long = df_proj.melt(
    id_vars=["SalaryUSD", "IsFemale", "SurveyYear"],
    value_vars=project_cols,
    value_name="ProjectType"
)

df_proj_long = df_proj_long.dropna(subset=["ProjectType"])
df_proj_long["ProjectType"] = df_proj_long["ProjectType"].str.strip()

# -------------------------
# Label formatting function (2 lines)
# -------------------------
def format_project_label(name):
    words = name.split()
    if len(words) > 2:
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    else:
        return name

# -------------------------
# Plot Function
# -------------------------
def plot_project_year(year):

    st.subheader(f"{year}")

    df_year = df_proj_long[df_proj_long["SurveyYear"] == year]

    gender_proj = (
        df_year
        .groupby(["ProjectType", "IsFemale"])["SalaryUSD"]
        .mean()
        .unstack()
    )

    gender_proj.columns = ["Men Avg Salary", "Women Avg Salary"]

    gender_proj["Gap %"] = (
        (gender_proj["Women Avg Salary"] - gender_proj["Men Avg Salary"])
        / gender_proj["Men Avg Salary"]
    ) * 100

    st.dataframe(gender_proj.round(2))

    fig, ax = plt.subplots(figsize=(16, 5))

    gender_proj[["Men Avg Salary", "Women Avg Salary"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Project Type")
    ax.set_title(f"Average Salary by Gender and Project Type ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Two-line labels
    formatted_labels = [format_project_label(p) for p in gender_proj.index]
    ax.set_xticklabels(formatted_labels, rotation=25, ha="right")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

plot_project_year(2015)
plot_project_year(2023)

# =========================
# CLEAN CERTIFICATION TYPES
# =========================

cert_columns = [
    "CertType1","CertType2","CertType3","CertType4",
    "CertType5","CertType6","CertType7","CertType8"
]

cert_codes = ["CCP","CCT","CEP","CFCC","CST","DRMP","EVP","PSP"]

# Create clean indicator columns for each certification
for cert in cert_codes:

    mask = False

    for col in cert_columns:
        mask |= df[col].astype(str).str.upper().str.startswith(cert)

    df[f"Has_{cert}"] = mask

st.header("Salary by Certification Type (2015 vs 2023)")

def cert_salary_by_year(year):

    df_year = df[df["SurveyYear"] == year]

    result = {}

    for cert in cert_codes:
        cert_df = df_year[df_year[f"Has_{cert}"] == True]

        if len(cert_df) > 5:
            result[cert] = cert_df["SalaryUSD"].mean()

    return pd.Series(result)


salary_2015 = cert_salary_by_year(2015)
salary_2023 = cert_salary_by_year(2023)

salary_compare = pd.concat([salary_2015, salary_2023], axis=1)
salary_compare.columns = ["2015 Avg Salary", "2023 Avg Salary"]

st.dataframe(salary_compare.round(0))

fig, ax = plt.subplots(figsize=(10, 4))
salary_compare.plot(kind="bar", ax=ax)

ax.set_ylabel("Average Salary (USD)")
ax.set_xlabel("Certification")
ax.set_title("Average Salary by Certification (2015 vs 2023)")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", fontsize=8)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("Salary by Certification and Gender (2015 vs 2023)")

def cert_gender_chart(year):

    st.subheader(f"{year}")

    df_year = df[df["SurveyYear"] == year]

    results = []

    for cert in cert_codes:

        cert_df = df_year[df_year[f"Has_{cert}"] == True]

        if len(cert_df) > 5:

            grouped = cert_df.groupby("IsFemale")["SalaryUSD"].mean()

            men_salary = grouped.get(False, 0)
            women_salary = grouped.get(True, 0)

            results.append([cert, men_salary, women_salary])

    cert_gender_df = pd.DataFrame(
        results,
        columns=["Certification", "Men Avg Salary", "Women Avg Salary"]
    )

    st.dataframe(cert_gender_df.round(0))

    fig, ax = plt.subplots(figsize=(10, 4))

    cert_gender_df.set_index("Certification")[
        ["Men Avg Salary", "Women Avg Salary"]
    ].plot(kind="bar", ax=ax)

    ax.set_ylabel("Average Salary (USD)")
    ax.set_xlabel("Certification")
    ax.set_title(f"Salary by Certification and Gender ({year})")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


cert_gender_chart(2015)
cert_gender_chart(2023)

# =========================
# ADVANCED MULTIVARIATE MODEL (FULL SAFE VERSION)
# =========================
st.header("Advanced Multivariate Salary Model")

df_reg = df.dropna(subset=[
    "SalaryUSD",
    "YearsOfExperience",
    "Age",
    "LevelOfEducation",
    "Industry"
]).copy()

# Clean education
df_reg["LevelOfEducation"] = df_reg["LevelOfEducation"].astype(str).str.strip().str.lower()

# Manager flag
df_reg["IsManager"] = df_reg["ManagerialDuties"].astype(str).str.contains("Yes", case=False, na=False)

# Consultant flag
df_reg["IsConsult"] = df_reg["Consult"].astype(str).str.contains("Yes", case=False, na=False)

# Convert booleans to int explicitly
df_reg["IsCertified"] = df_reg["IsCertified"].astype(int)
df_reg["IsMember"] = df_reg["IsMember"].astype(int)
df_reg["IsFemale"] = df_reg["IsFemale"].astype(int)
df_reg["IsManager"] = df_reg["IsManager"].astype(int)
df_reg["IsConsult"] = df_reg["IsConsult"].astype(int)

# Build base X
X = df_reg[
    [
        "YearsOfExperience",
        "Age",
        "IsCertified",
        "IsMember",
        "IsFemale",
        "IsManager",
        "IsConsult"
    ]
].copy()

# Convert numeric safely
X["YearsOfExperience"] = pd.to_numeric(X["YearsOfExperience"], errors="coerce")
X["Age"] = pd.to_numeric(X["Age"], errors="coerce")

# Dummy variables (force numeric)
edu_dummies = pd.get_dummies(df_reg["LevelOfEducation"], drop_first=True).astype(int)
ind_dummies = pd.get_dummies(df_reg["Industry"], drop_first=True).astype(int)

# Combine everything
X = pd.concat([X, edu_dummies, ind_dummies], axis=1)

# Remove any leftover NaNs
X = X.dropna()
y = pd.to_numeric(df_reg.loc[X.index, "SalaryUSD"], errors="coerce")

# Add constant
X = sm.add_constant(X)

# FINAL numeric check
X = X.astype(float)
y = y.astype(float)

model = sm.OLS(y, X).fit()

# =========================
# CLEAN REGRESSION TABLE
# =========================
st.subheader("Regression Results (Clean Summary)")

results_df = pd.DataFrame({
    "Variable": model.params.index,
    "Coefficient (USD)": model.params.values,
    "Std Error": model.bse.values,
    "P-Value": model.pvalues.values
})

# Remove constant
results_df = results_df[results_df["Variable"] != "const"]

# Round values
results_df["Coefficient (USD)"] = results_df["Coefficient (USD)"].round(0)
results_df["Std Error"] = results_df["Std Error"].round(0)
results_df["P-Value"] = results_df["P-Value"].round(4)

# Add significance column
results_df["Significant?"] = results_df["P-Value"] < 0.05

# Optional: Keep only main variables
main_vars = [
    "YearsOfExperience",
    "Age",
    "IsCertified",
    "IsMember",
    "IsFemale",
    "IsManager",
    "IsConsult"
]

results_main = results_df[results_df["Variable"].isin(main_vars)]

st.dataframe(results_main)


# =========================
# VISUAL: KEY COEFFICIENTS
# =========================
st.subheader("Key Salary Impact Factors (Visual)")

coef_df = pd.DataFrame({
    "Variable": model.params.index,
    "Coefficient": model.params.values,
    "p_value": model.pvalues.values
})

# Keep only main variables
main_vars = [
    "YearsOfExperience",
    "Age",
    "IsCertified",
    "IsMember",
    "IsFemale",
    "IsManager",
    "IsConsult"
]

coef_df = coef_df[coef_df["Variable"].isin(main_vars)]

# Sort by impact
coef_df = coef_df.sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.barh(coef_df["Variable"], coef_df["Coefficient"])

ax.set_title("Impact on Salary (Holding Other Factors Constant)")
ax.set_xlabel("Salary Impact (USD)")
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Add data labels
for i, v in enumerate(coef_df["Coefficient"]):
    ax.text(v, i, f"{v:,.0f}", va='center', fontsize=8)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


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
# =========================
# CLEAN REGRESSION TABLE
# =========================
st.subheader("Regression Results (Simplified Model)")

results_df = pd.DataFrame({
    "Variable": model.params.index,
    "Coefficient (USD)": model.params.values,
    "Std Error": model.bse.values,
    "P-Value": model.pvalues.values
})

# Remove constant
results_df = results_df[results_df["Variable"] != "const"]

# Round nicely
results_df["Coefficient (USD)"] = results_df["Coefficient (USD)"].round(0)
results_df["Std Error"] = results_df["Std Error"].round(0)
results_df["P-Value"] = results_df["P-Value"].round(4)

# Add significance flag
results_df["Statistically Significant (p < 0.05)"] = results_df["P-Value"] < 0.05

st.dataframe(results_df)

# =========================
# DEBUG: SHOW ALL COLUMNS
# =========================
st.header("Debug: Column Names")

st.write("All Columns in Dataset:")
st.write(df.columns.tolist())
