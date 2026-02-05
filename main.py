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
