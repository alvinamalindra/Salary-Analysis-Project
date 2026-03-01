import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import omni_normtest, jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

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

# =========================
# TAB NAVIGATION
# =========================
tab_overview, tab_histogram, tab_satisfaction, tab_gender, tab_certification, tab_regression, tab_causation = st.tabs([
    "Overview",
    "Histograms",
    "Job Satisfaction",
    "Gender Gap",
    "Certification",
    "Regression Model",
    "Causation Analysis"
])

with tab_overview:
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

with tab_histogram:
    # =========================
    # HISTOGRAM ANALYSIS
    # =========================
    st.header("Salary Distribution — Histogram Analysis")
    st.markdown("Salary histograms in **$5,000 buckets** to visualize the shape, spread, and kurtosis of the dataset. Includes **mean, median, and mode** lines.")

    def plot_histogram(df_input, title, color="steelblue", bucket_size=5000):
        """Plot a salary histogram with mean, median, mode lines."""
        df_plot = df_input.dropna(subset=["SalaryUSD"]).copy()
        df_plot = df_plot[df_plot["SalaryUSD"].between(5000, 500000)]

        if len(df_plot) < 10:
            st.warning(f"Not enough data for: {title}")
            return

        salaries = df_plot["SalaryUSD"]

        mean_sal = salaries.mean()
        median_sal = salaries.median()
        std_sal = salaries.std()
        skewness = stats.skew(salaries, nan_policy="omit")
        kurt = stats.kurtosis(salaries, nan_policy="omit")
        n = len(salaries)

        # Mode: midpoint of most frequent bin
        bin_min_pre = int(np.floor(salaries.min() / bucket_size) * bucket_size)
        bin_max_pre = int(np.ceil(salaries.max() / bucket_size) * bucket_size) + bucket_size
        bins_pre = np.arange(bin_min_pre, bin_max_pre, bucket_size)
        counts_pre, edges_pre = np.histogram(salaries, bins=bins_pre)
        mode_bin_idx = np.argmax(counts_pre)
        mode_sal = (edges_pre[mode_bin_idx] + edges_pre[mode_bin_idx + 1]) / 2

        bins = np.arange(bin_min_pre, bin_max_pre, bucket_size)

        fig, ax = plt.subplots(figsize=(12, 5))
        counts, edges, patches = ax.hist(salaries, bins=bins, color=color, edgecolor="white", alpha=0.85, linewidth=0.6)

        ax.axvline(mean_sal, color="red", linewidth=2, linestyle="--", label=f"Mean: ${mean_sal:,.0f}")
        ax.axvline(median_sal, color="orange", linewidth=2, linestyle="-.", label=f"Median: ${median_sal:,.0f}")
        ax.axvline(mode_sal, color="green", linewidth=2, linestyle=":", label=f"Mode: ${mode_sal:,.0f}")
        ax.axvspan(mean_sal - std_sal, mean_sal + std_sal, alpha=0.08, color="red",
                   label=f"\u00b11\u03c3 (${mean_sal - std_sal:,.0f} \u2013 ${mean_sal + std_sal:,.0f})")

        ax.set_xlabel("Salary (USD)", fontsize=11)
        ax.set_ylabel("Number of Respondents", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.ticklabel_format(style="plain", axis="x")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        stats_data = pd.DataFrame({
            "Metric": ["N (Count)", "Mean Salary", "Median Salary", "Mode Salary (bin midpoint)",
                       "Std Deviation", "Skewness", "Excess Kurtosis", "Min Salary", "Max Salary",
                       "25th Percentile (Q1)", "75th Percentile (Q3)", "IQR"],
            "Value": [f"{n:,}", f"${mean_sal:,.0f}", f"${median_sal:,.0f}", f"${mode_sal:,.0f}",
                      f"${std_sal:,.0f}", f"{skewness:.4f}", f"{kurt:.4f}",
                      f"${salaries.min():,.0f}", f"${salaries.max():,.0f}",
                      f"${salaries.quantile(0.25):,.0f}", f"${salaries.quantile(0.75):,.0f}",
                      f"${salaries.quantile(0.75) - salaries.quantile(0.25):,.0f}"]
        })
        st.dataframe(stats_data, hide_index=True)

        if skewness > 0.5:
            skew_text = "**Right-skewed** (positively skewed) — a tail of high earners pulls the distribution rightward."
        elif skewness < -0.5:
            skew_text = "**Left-skewed** (negatively skewed) — a concentration at higher salaries with a tail of lower earners."
        else:
            skew_text = "**Approximately symmetric** — salaries are fairly evenly distributed around the center."

        if kurt > 1:
            kurt_text = "**Leptokurtic** (heavy tails) — more extreme salary values than a normal distribution."
        elif kurt < -1:
            kurt_text = "**Platykurtic** (light tails) — fewer extreme salary values than a normal distribution."
        else:
            kurt_text = "**Mesokurtic** (near-normal) — the tail behavior is similar to a normal distribution."

        st.markdown(f"""
    **Shape Analysis:**
    - Skewness ({skewness:.4f}): {skew_text}
    - Kurtosis ({kurt:.4f}): {kurt_text}
    """)

    # --- All Respondents ---
    st.subheader("All Respondents")
    plot_histogram(df, "Salary Distribution — All Respondents", color="steelblue")

    # --- Men vs Women ---
    st.subheader("Men vs Women")
    df_hist_men = df[~df["IsFemale"]]
    df_hist_women = df[df["IsFemale"]]

    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df_hist_men, "Salary Distribution — Men", color="#3A86FF")
    with c2:
        plot_histogram(df_hist_women, "Salary Distribution — Women", color="#FF006E")

    # --- 2015 vs 2023 ---
    st.subheader("Year-over-Year Comparison (2015 vs 2023)")

    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df_2015, "All Respondents — 2015", color="#457B9D")
    with c2:
        plot_histogram(df_2023, "All Respondents — 2023", color="#E63946")

    # --- Overlay: Men vs Women ---
    st.subheader("Overlay: Men vs Women")

    df_men_sal = df_hist_men["SalaryUSD"].dropna()
    df_women_sal = df_hist_women["SalaryUSD"].dropna()
    df_men_sal = df_men_sal[df_men_sal.between(5000, 500000)]
    df_women_sal = df_women_sal[df_women_sal.between(5000, 500000)]

    if len(df_men_sal) > 10 and len(df_women_sal) > 10:
        bucket_size = 5000
        all_salaries = pd.concat([df_men_sal, df_women_sal])
        bin_min = int(np.floor(all_salaries.min() / bucket_size) * bucket_size)
        bin_max = int(np.ceil(all_salaries.max() / bucket_size) * bucket_size) + bucket_size
        bins = np.arange(bin_min, bin_max, bucket_size)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.hist(df_men_sal, bins=bins, alpha=0.55, color="#3A86FF", edgecolor="white", linewidth=0.5, label=f"Men (n={len(df_men_sal):,})")
        ax.hist(df_women_sal, bins=bins, alpha=0.55, color="#FF006E", edgecolor="white", linewidth=0.5, label=f"Women (n={len(df_women_sal):,})")

        men_counts, men_edges = np.histogram(df_men_sal, bins=bins)
        men_mode_idx = np.argmax(men_counts)
        men_mode = (men_edges[men_mode_idx] + men_edges[men_mode_idx + 1]) / 2

        women_counts, women_edges = np.histogram(df_women_sal, bins=bins)
        women_mode_idx = np.argmax(women_counts)
        women_mode = (women_edges[women_mode_idx] + women_edges[women_mode_idx + 1]) / 2

        ax.axvline(df_men_sal.mean(), color="#3A86FF", linewidth=2, linestyle="--", label=f"Men Mean: ${df_men_sal.mean():,.0f}")
        ax.axvline(df_women_sal.mean(), color="#FF006E", linewidth=2, linestyle="--", label=f"Women Mean: ${df_women_sal.mean():,.0f}")
        ax.axvline(df_men_sal.median(), color="#3A86FF", linewidth=2, linestyle="-.", label=f"Men Median: ${df_men_sal.median():,.0f}")
        ax.axvline(df_women_sal.median(), color="#FF006E", linewidth=2, linestyle="-.", label=f"Women Median: ${df_women_sal.median():,.0f}")
        ax.axvline(men_mode, color="#3A86FF", linewidth=2, linestyle=":", label=f"Men Mode: ${men_mode:,.0f}")
        ax.axvline(women_mode, color="#FF006E", linewidth=2, linestyle=":", label=f"Women Mode: ${women_mode:,.0f}")

        ax.set_xlabel("Salary (USD)", fontsize=11)
        ax.set_ylabel("Number of Respondents", fontsize=11)
        ax.set_title("Overlaid Salary Distribution — Men vs Women", fontsize=13, fontweight="bold")
        ax.ticklabel_format(style="plain", axis="x")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        gap = df_men_sal.mean() - df_women_sal.mean()
        gap_pct = (gap / df_men_sal.mean()) * 100

        summary = pd.DataFrame({
            "Metric": ["Count", "Mean", "Median", "Mode (bin midpoint)", "Std Dev", "Skewness", "Excess Kurtosis"],
            "Men": [f"{len(df_men_sal):,}", f"${df_men_sal.mean():,.0f}", f"${df_men_sal.median():,.0f}",
                    f"${men_mode:,.0f}", f"${df_men_sal.std():,.0f}",
                    f"{stats.skew(df_men_sal):.4f}", f"{stats.kurtosis(df_men_sal):.4f}"],
            "Women": [f"{len(df_women_sal):,}", f"${df_women_sal.mean():,.0f}", f"${df_women_sal.median():,.0f}",
                      f"${women_mode:,.0f}", f"${df_women_sal.std():,.0f}",
                      f"{stats.skew(df_women_sal):.4f}", f"{stats.kurtosis(df_women_sal):.4f}"]
        })
        st.dataframe(summary, hide_index=True)

        st.markdown(f"""
    **Gender Pay Gap Summary:**
    - Mean salary gap: **${gap:,.0f}** ({gap_pct:.1f}% lower for women)
    - Men mean: ${df_men_sal.mean():,.0f} | Women mean: ${df_women_sal.mean():,.0f}
    """)
    else:
        st.warning("Not enough data for Men or Women to create the overlay chart.")


with tab_satisfaction:
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
    st.pyplot(fig2)
    plt.close(fig2)

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

with tab_gender:
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

with tab_certification:
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

with tab_regression:
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
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_stat = model.fvalue
    f_pvalue = model.f_pvalue

    # =========================
    # MODEL PERFORMANCE METRICS
    # =========================
    st.subheader("Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("R²", f"{r_squared:.3f}")
    col2.metric("Adjusted R²", f"{adj_r_squared:.3f}")
    col3.metric("F-Statistic", f"{f_stat:.1f}")
    col4.metric("Model p-value", f"{f_pvalue:.4e}")

    # =========================
    # VISUAL: MAIN IMPACT FACTORS ONLY
    # =========================
    st.subheader("Key Salary Impact Factors (Core Variables Only)")

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    # Remove constant
    coef_df = coef_df[coef_df["Variable"] != "const"]

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

    # Sort by absolute impact
    coef_df["AbsImpact"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsImpact", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(coef_df["Variable"], coef_df["Coefficient"])

    ax.set_title("Impact on Salary (Core Variables)")
    ax.set_xlabel("Salary Impact (USD)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(coef_df["Coefficient"]):
        ax.text(v, i, f"{v:,.0f}", va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # REGRESSION RESULTS (RANKED BY COEFFICIENT)
    # =========================
    st.subheader("Regression Results")

    results_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient (USD)": model.params.values,
        "Std Error": model.bse.values,
        "P-Value": model.pvalues.values
    })

    # Remove constant
    results_df = results_df[results_df["Variable"] != "const"]

    # Keep only core variables
    main_vars = [
        "YearsOfExperience",
        "Age",
        "IsCertified",
        "IsMember",
        "IsFemale",
        "IsManager",
        "IsConsult"
    ]

    results_df = results_df[results_df["Variable"].isin(main_vars)]

    # Add significance flag
    results_df["Significant"] = results_df["P-Value"].apply(
        lambda x: "✅" if x < 0.05 else "❌"
    )

    # SORT BY COEFFICIENT (descending)
    results_df = results_df.sort_values(
        by="Coefficient (USD)",
        ascending=False
    )

    # Clean formatting
    results_df["Coefficient (USD)"] = results_df["Coefficient (USD)"].round(0)
    results_df["Std Error"] = results_df["Std Error"].round(0)
    results_df["P-Value"] = results_df["P-Value"].round(4)

    st.dataframe(results_df)



    # =========================
    # MODEL DIAGNOSTICS
    # =========================
    st.subheader("Model Diagnostics")

    # Residuals
    residuals = model.resid

    # Compute tests manually
    omni_stat, omni_p = omni_normtest(residuals)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    dw_stat = durbin_watson(residuals)

    diagnostics_df = pd.DataFrame({
        "Metric": [
            "Omnibus",
            "Prob(Omnibus)",
            "Jarque-Bera",
            "Prob(JB)",
            "Skew",
            "Kurtosis",
            "Durbin-Watson",
            "Condition Number"
        ],
        "Value": [
            omni_stat,
            omni_p,
            jb_stat,
            jb_p,
            skew,
            kurtosis,
            dw_stat,
            model.condition_number
        ]
    })

    st.dataframe(diagnostics_df.round(3))

    # =========================
    # VIF ANALYSIS
    # =========================
    st.subheader("VIF (Variance Inflation Factor) Analysis")
    st.markdown("""
    **Why VIF?** The Condition Number of **{:.0f}** indicates severe multicollinearity.
    VIF measures how much each variable's variance is inflated due to correlation with other predictors.
    - **VIF < 5** → ✅ Acceptable
    - **VIF 5–10** → ⚠️ Moderate concern
    - **VIF > 10** → 🔴 High multicollinearity — action needed
    """.format(model.condition_number))

    # Compute VIF for each predictor (skip constant at index 0)
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    # Remove constant row for display
    vif_data = vif_data[vif_data["Variable"] != "const"]

    # Sort by VIF descending
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    # Add interpretation flag
    def vif_flag(v):
        if v > 10:
            return "🔴 High"
        elif v >= 5:
            return "⚠️ Moderate"
        else:
            return "✅ OK"

    vif_data["Status"] = vif_data["VIF"].apply(vif_flag)
    vif_data["VIF"] = vif_data["VIF"].round(2)

    st.dataframe(vif_data)

    # Identify high-VIF variables (VIF > 10, excluding dummies)
    high_vif_vars = vif_data[
        (vif_data["VIF"] > 10) &
        (vif_data["Variable"].isin(main_vars))
    ]["Variable"].tolist()

    # Also identify moderate-VIF variables (5-10)
    moderate_vif_vars = vif_data[
        (vif_data["VIF"] >= 5) & (vif_data["VIF"] <= 10) &
        (vif_data["Variable"].isin(main_vars))
    ]["Variable"].tolist()

    if high_vif_vars:
        st.warning(f"⚠️ High multicollinearity detected in: **{', '.join(high_vif_vars)}**")
    elif moderate_vif_vars:
        st.warning(f"⚠️ Moderate multicollinearity detected in: **{', '.join(moderate_vif_vars)}**. "
                   f"Combined with the intercept and dummy variables, this contributes to the high Condition Number.")
    else:
        st.success("✅ No multicollinearity among core variables.")



with tab_causation:
    # =========================
    # FIXED MODEL (MULTICOLLINEARITY REMEDIATION)
    # =========================
    st.header("Fixed Model — Multicollinearity Remediation")

    # Determine remediation strategy
    vars_to_drop = []
    apply_centering = False

    # Check if Age and YearsOfExperience are both at least moderate-VIF
    age_yoe_correlated = (
        ("Age" in high_vif_vars or "Age" in moderate_vif_vars) and
        ("YearsOfExperience" in high_vif_vars or "YearsOfExperience" in moderate_vif_vars)
    )

    if age_yoe_correlated:
        vars_to_drop = ["Age"]
        apply_centering = True
        st.markdown("""
    **Fixes Applied:**
    1. **Drop `Age`** — `Age` and `YearsOfExperience` have moderate VIF (~5.4 each), confirming
       they carry redundant information. Keeping `YearsOfExperience` because it is more directly
       relevant to salary determination.
    2. **Mean-center `YearsOfExperience`** — Centering reduces the artificial inflation of the
       Condition Number caused by the interaction between continuous variables and the intercept/dummy
       structure.
    """)
    elif high_vif_vars:
        vars_to_drop = high_vif_vars[:1]
        apply_centering = True
        st.markdown(f"**Fix Applied:** Dropping `{vars_to_drop[0]}` (highest VIF) and mean-centering continuous variables.")
    else:
        apply_centering = True
        st.info("No core variables have concerning VIF. Applying mean-centering to reduce Condition Number from dummy/intercept structure.")

    # Build fixed model
    X_fixed = X.drop(columns=vars_to_drop, errors="ignore").copy()

    if apply_centering:
        for col in ["YearsOfExperience", "Age"]:
            if col in X_fixed.columns:
                X_fixed[col] = X_fixed[col] - X_fixed[col].mean()

    y_fixed = y.loc[X_fixed.index]

    model_fixed = sm.OLS(y_fixed, X_fixed).fit()

    # =========================
    # FIXED MODEL PERFORMANCE
    # =========================
    st.subheader("Fixed Model — Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{model_fixed.rsquared:.3f}")
    col2.metric("Adjusted R²", f"{model_fixed.rsquared_adj:.3f}")
    col3.metric("F-Statistic", f"{model_fixed.fvalue:.1f}")
    col4.metric("Model p-value", f"{model_fixed.f_pvalue:.4e}")

    # =========================
    # FIXED MODEL KEY IMPACT FACTORS
    # =========================
    st.subheader("Fixed Model — Key Salary Impact Factors")

    coef_fixed = pd.DataFrame({
        "Variable": model_fixed.params.index,
        "Coefficient": model_fixed.params.values,
        "p_value": model_fixed.pvalues.values
    })

    coef_fixed = coef_fixed[coef_fixed["Variable"] != "const"]

    fixed_main_vars = [v for v in main_vars if v not in vars_to_drop]
    coef_fixed = coef_fixed[coef_fixed["Variable"].isin(fixed_main_vars)]

    coef_fixed["AbsImpact"] = coef_fixed["Coefficient"].abs()
    coef_fixed = coef_fixed.sort_values("AbsImpact", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(coef_fixed["Variable"], coef_fixed["Coefficient"])
    ax.set_title("Impact on Salary — Fixed Model (Core Variables)")
    ax.set_xlabel("Salary Impact (USD)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(coef_fixed["Coefficient"]):
        ax.text(v, i, f"{v:,.0f}", va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # FIXED MODEL REGRESSION TABLE
    # =========================
    st.subheader("Fixed Model — Regression Results")

    results_fixed = pd.DataFrame({
        "Variable": model_fixed.params.index,
        "Coefficient (USD)": model_fixed.params.values,
        "Std Error": model_fixed.bse.values,
        "P-Value": model_fixed.pvalues.values
    })

    results_fixed = results_fixed[results_fixed["Variable"] != "const"]
    results_fixed = results_fixed[results_fixed["Variable"].isin(fixed_main_vars)]

    results_fixed["Significant"] = results_fixed["P-Value"].apply(
        lambda x: "✅" if x < 0.05 else "❌"
    )

    results_fixed = results_fixed.sort_values(by="Coefficient (USD)", ascending=False)
    results_fixed["Coefficient (USD)"] = results_fixed["Coefficient (USD)"].round(0)
    results_fixed["Std Error"] = results_fixed["Std Error"].round(0)
    results_fixed["P-Value"] = results_fixed["P-Value"].round(4)

    st.dataframe(results_fixed)

    # =========================
    # FIXED MODEL DIAGNOSTICS
    # =========================
    st.subheader("Fixed Model — Diagnostics")

    residuals_fixed = model_fixed.resid
    omni_stat_f, omni_p_f = omni_normtest(residuals_fixed)
    jb_stat_f, jb_p_f, skew_f, kurtosis_f = jarque_bera(residuals_fixed)
    dw_stat_f = durbin_watson(residuals_fixed)

    diag_fixed = pd.DataFrame({
        "Metric": [
            "Omnibus", "Prob(Omnibus)",
            "Jarque-Bera", "Prob(JB)",
            "Skew", "Kurtosis",
            "Durbin-Watson", "Condition Number"
        ],
        "Value": [
            omni_stat_f, omni_p_f,
            jb_stat_f, jb_p_f,
            skew_f, kurtosis_f,
            dw_stat_f, model_fixed.condition_number
        ]
    })

    st.dataframe(diag_fixed.round(3))

    # =========================
    # VIF ON FIXED MODEL
    # =========================
    st.subheader("Fixed Model — VIF Check")

    vif_fixed = pd.DataFrame({
        "Variable": X_fixed.columns,
        "VIF": [variance_inflation_factor(X_fixed.values, i) for i in range(X_fixed.shape[1])]
    })
    vif_fixed = vif_fixed[vif_fixed["Variable"] != "const"]
    vif_fixed = vif_fixed.sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_fixed["Status"] = vif_fixed["VIF"].apply(vif_flag)
    vif_fixed["VIF"] = vif_fixed["VIF"].round(2)

    st.dataframe(vif_fixed)

    # =========================
    # SIDE-BY-SIDE COMPARISON
    # =========================
    st.header("Model Comparison: Original vs Fixed")

    comparison_df = pd.DataFrame({
        "Metric": [
            "R²",
            "Adjusted R²",
            "F-Statistic",
            "Condition Number",
            "Durbin-Watson",
            "Skew",
            "Kurtosis"
        ],
        "Original Model": [
            round(model.rsquared, 4),
            round(model.rsquared_adj, 4),
            round(model.fvalue, 1),
            round(model.condition_number, 1),
            round(dw_stat, 3),
            round(skew, 3),
            round(kurtosis, 3)
        ],
        "Fixed Model": [
            round(model_fixed.rsquared, 4),
            round(model_fixed.rsquared_adj, 4),
            round(model_fixed.fvalue, 1),
            round(model_fixed.condition_number, 1),
            round(dw_stat_f, 3),
            round(skew_f, 3),
            round(kurtosis_f, 3)
        ]
    })

    st.dataframe(comparison_df)

    # Interpretation
    cond_original = model.condition_number
    cond_fixed = model_fixed.condition_number
    improvement = ((cond_original - cond_fixed) / cond_original) * 100

    if cond_fixed < 30:
        st.success(f"""✅ **Multicollinearity resolved!**
    - Condition Number dropped from **{cond_original:,.0f}** → **{cond_fixed:,.0f}** ({improvement:.1f}% reduction)
    - The fixed model is now safe for causal interpretation.""")
    elif cond_fixed < 100:
        st.warning(f"""⚠️ **Multicollinearity improved but still elevated.**
    - Condition Number: **{cond_original:,.0f}** → **{cond_fixed:,.0f}** ({improvement:.1f}% reduction)
    - Consider Ridge Regression or further variable reduction.""")
    else:
        st.error(f"""🔴 **Multicollinearity still present.**
    - Condition Number: **{cond_original:,.0f}** → **{cond_fixed:,.0f}** ({improvement:.1f}% reduction)
    - Recommend adding more control variables to improve the model.""")


    # ============================================================


    # ============================================================
    # PHASE 2: ENHANCED CAUSATION MODEL
    # ============================================================
    st.header("Enhanced Causation Model")
    st.markdown("""
    This model adds **more control variables** to reduce omitted variable bias and
    strengthen causal claims. Based on econometric research (Wilms et al., 2021;
    Cinelli & Hazlett, 2020), adding relevant controls is the proper way to improve
    causal inference — not switching to Ridge Regression (which biases coefficients).
    """)

    # =========================
    # STEP 1: CLEAN & ENCODE NEW VARIABLES
    # =========================

    # --- Region Grouping from LocationWork ---
    df_enhanced = df.dropna(subset=[
        "SalaryUSD", "YearsOfExperience",
        "LocationWork", "WorkFunction", "ProjectSize"
    ]).copy()

    def assign_region(loc):
        loc_upper = str(loc).upper()
        if "UNITED STATES" in loc_upper or "USA" in loc_upper:
            return "US"
        elif "CANADA" in loc_upper:
            return "Canada"
        elif any(x in loc_upper for x in [
            "UNITED ARAB", "SAUDI", "QATAR", "KUWAIT", "OMAN", "BAHRAIN", "IRAQ", "JORDAN", "LEBANON"
        ]):
            return "Middle East"
        elif any(x in loc_upper for x in [
            "UNITED KINGDOM", "GERMANY", "FRANCE", "NETHERLANDS", "SPAIN", "ITALY",
            "NORWAY", "SWEDEN", "SWITZERLAND", "BELGIUM", "IRELAND", "AUSTRIA",
            "DENMARK", "FINLAND", "PORTUGAL", "POLAND", "CZECH", "ROMANIA", "EUROPE"
        ]):
            return "Europe"
        elif any(x in loc_upper for x in [
            "AUSTRALIA", "INDIA", "CHINA", "JAPAN", "SINGAPORE", "MALAYSIA",
            "INDONESIA", "PHILIPPINES", "KOREA", "THAILAND", "VIETNAM",
            "NEW ZEALAND", "PAKISTAN", "BANGLADESH", "HONG KONG", "TAIWAN"
        ]):
            return "Asia-Pacific"
        else:
            return "Other"

    df_enhanced["Region"] = df_enhanced["LocationWork"].apply(assign_region)

    # --- Clean WorkFunction ---
    df_enhanced["WorkFunction"] = df_enhanced["WorkFunction"].astype(str).str.strip()
    df_enhanced["WorkFunction"] = df_enhanced["WorkFunction"].replace({
        "Other (please specify)": "Other"
    })

    # --- Standardize ProjectSize ---
    def standardize_project_size(ps):
        ps_str = str(ps).upper().replace(",", "").strip()
        if any(x in ps_str for x in ["0-5", "0 - 5"]):
            return "0-5M"
        elif any(x in ps_str for x in ["5-20", "5 - 20"]):
            return "5-20M"
        elif any(x in ps_str for x in ["20-100", "20 - 100"]):
            return "20-100M"
        elif any(x in ps_str for x in ["100-500", "100 - 500"]):
            return "100-500M"
        elif any(x in ps_str for x in ["500-1000", "500 - 1000"]):
            return "500M-1B"
        elif "1000" in ps_str:
            return "1B+"
        else:
            return ps_str

    df_enhanced["ProjectSizeClean"] = df_enhanced["ProjectSize"].apply(standardize_project_size)

    # --- Numeric columns ---
    df_enhanced["YearsOfExperience"] = pd.to_numeric(df_enhanced["YearsOfExperience"], errors="coerce")
    df_enhanced["SalaryUSD"] = pd.to_numeric(df_enhanced["SalaryUSD"], errors="coerce")
    df_enhanced["WorkHours"] = pd.to_numeric(df_enhanced["WorkHours"], errors="coerce")
    df_enhanced["YrsWithEmployer"] = pd.to_numeric(df_enhanced["YrsWithEmployer"], errors="coerce")
    df_enhanced["CompanySize"] = pd.to_numeric(df_enhanced["NumberOfEmployeesInCompany"], errors="coerce")
    df_enhanced["LogCompanySize"] = np.log1p(df_enhanced["CompanySize"])

    # --- Boolean flags (to int) ---
    df_enhanced["IsCertified"] = df_enhanced["IsCertified"].astype(int)
    df_enhanced["IsMember"] = df_enhanced["IsMember"].astype(int)
    df_enhanced["IsFemale"] = df_enhanced["IsFemale"].astype(int)
    df_enhanced["IsManager"] = df_enhanced["ManagerialDuties"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_enhanced["IsConsult"] = df_enhanced["Consult"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_enhanced["HasPE"] = df_enhanced["PE"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_enhanced["HasTechDegree"] = df_enhanced["TechnicalDegree"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_enhanced["HasBizDegree"] = df_enhanced["BusinessDegree"].astype(str).str.contains("Yes", case=False, na=False).astype(int)

    # --- Education grouping (5 → 3 levels) ---
    df_enhanced["LevelOfEducation"] = df_enhanced["LevelOfEducation"].astype(str).str.strip().str.lower()
    def group_education(edu):
        if any(x in edu for x in ["high school", "associate"]):
            return "Low"
        elif any(x in edu for x in ["undergraduate", "bachelor"]):
            return "Mid"
        elif any(x in edu for x in ["graduate", "master", "doctoral"]):
            return "High"
        else:
            return "Mid"

    df_enhanced["EduGroup"] = df_enhanced["LevelOfEducation"].apply(group_education)

    # =========================
    # STEP 2: BUILD ENHANCED MODEL
    # =========================
    st.subheader("Variables Added")

    region_counts = df_enhanced["Region"].value_counts()
    st.markdown("**LocationWork → Region Grouping:**")
    st.dataframe(region_counts.reset_index().rename(columns={"index": "Region", "Region": "Region", "count": "Count"}))

    # Build X matrix
    enhanced_core_vars = [
        "YearsOfExperience",
        "IsCertified", "IsMember", "IsFemale",
        "IsManager", "IsConsult",
        "HasPE", "HasTechDegree", "HasBizDegree",
        "LogCompanySize", "WorkHours"
    ]

    # Add YrsWithEmployer only if enough data
    if df_enhanced["YrsWithEmployer"].notna().sum() > len(df_enhanced) * 0.5:
        enhanced_core_vars.append("YrsWithEmployer")

    X_enh = df_enhanced[enhanced_core_vars].copy()

    # Convert numeric safely
    for col in X_enh.columns:
        X_enh[col] = pd.to_numeric(X_enh[col], errors="coerce")

    # Add dummies
    region_dummies = pd.get_dummies(df_enhanced["Region"], drop_first=True, prefix="Region").astype(int)
    func_dummies = pd.get_dummies(df_enhanced["WorkFunction"], drop_first=True, prefix="Func").astype(int)
    proj_dummies = pd.get_dummies(df_enhanced["ProjectSizeClean"], drop_first=True, prefix="ProjSize").astype(int)
    edu_dummies_enh = pd.get_dummies(df_enhanced["EduGroup"], drop_first=True, prefix="Edu").astype(int)

    X_enh = pd.concat([X_enh, region_dummies, func_dummies, proj_dummies, edu_dummies_enh], axis=1)

    # Clean
    X_enh = X_enh.dropna()
    y_enh = pd.to_numeric(df_enhanced.loc[X_enh.index, "SalaryUSD"], errors="coerce")

    # Center YearsOfExperience
    X_enh["YearsOfExperience"] = X_enh["YearsOfExperience"] - X_enh["YearsOfExperience"].mean()

    # Add constant
    X_enh = sm.add_constant(X_enh)
    X_enh = X_enh.astype(float)
    y_enh = y_enh.astype(float)

    # Fit
    model_enhanced = sm.OLS(y_enh, X_enh).fit()

    # =========================
    # ENHANCED MODEL PERFORMANCE
    # =========================
    st.subheader("Enhanced Model — Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{model_enhanced.rsquared:.3f}")
    col2.metric("Adjusted R²", f"{model_enhanced.rsquared_adj:.3f}")
    col3.metric("F-Statistic", f"{model_enhanced.fvalue:.1f}")
    col4.metric("Model p-value", f"{model_enhanced.f_pvalue:.4e}")

    st.markdown(f"**Observations:** {int(model_enhanced.nobs)}")

    # =========================
    # ENHANCED MODEL KEY IMPACT FACTORS
    # =========================
    st.subheader("Enhanced Model — Key Salary Impact Factors")

    coef_enh = pd.DataFrame({
        "Variable": model_enhanced.params.index,
        "Coefficient": model_enhanced.params.values,
        "p_value": model_enhanced.pvalues.values
    })

    coef_enh = coef_enh[coef_enh["Variable"] != "const"]

    enhanced_display_vars = [
        "YearsOfExperience",
        "IsCertified", "IsMember", "IsFemale",
        "IsManager", "IsConsult",
        "HasPE", "HasTechDegree", "HasBizDegree",
        "LogCompanySize", "WorkHours"
    ]
    if "YrsWithEmployer" in enhanced_core_vars:
        enhanced_display_vars.append("YrsWithEmployer")

    coef_enh_main = coef_enh[coef_enh["Variable"].isin(enhanced_display_vars)].copy()
    coef_enh_main["AbsImpact"] = coef_enh_main["Coefficient"].abs()
    coef_enh_main = coef_enh_main.sort_values("AbsImpact", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in coef_enh_main["p_value"]]
    ax.barh(coef_enh_main["Variable"], coef_enh_main["Coefficient"], color=colors)
    ax.set_title("Impact on Salary — Enhanced Model (Core Variables)")
    ax.set_xlabel("Salary Impact (USD)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (v, p) in enumerate(zip(coef_enh_main["Coefficient"], coef_enh_main["p_value"])):
        label = f"{v:,.0f}" + (" *" if p < 0.05 else "")
        ax.text(v, i, label, va='center', fontsize=8)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#2ecc71", lw=6, label="Significant (p < 0.05)"),
            plt.Line2D([0], [0], color="#95a5a6", lw=6, label="Not significant")
        ],
        loc="lower right", fontsize=8
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # ENHANCED MODEL REGRESSION TABLE
    # =========================
    st.subheader("Enhanced Model — Regression Results (Core Variables)")

    results_enh = pd.DataFrame({
        "Variable": model_enhanced.params.index,
        "Coefficient (USD)": model_enhanced.params.values,
        "Std Error": model_enhanced.bse.values,
        "P-Value": model_enhanced.pvalues.values
    })

    results_enh = results_enh[results_enh["Variable"] != "const"]
    results_enh = results_enh[results_enh["Variable"].isin(enhanced_display_vars)]

    results_enh["Significant"] = results_enh["P-Value"].apply(
        lambda x: "✅" if x < 0.05 else "❌"
    )

    results_enh = results_enh.sort_values(by="Coefficient (USD)", ascending=False)
    results_enh["Coefficient (USD)"] = results_enh["Coefficient (USD)"].round(0)
    results_enh["Std Error"] = results_enh["Std Error"].round(0)
    results_enh["P-Value"] = results_enh["P-Value"].round(4)

    st.dataframe(results_enh)

    # =========================
    # ENHANCED MODEL DIAGNOSTICS
    # =========================
    st.subheader("Enhanced Model — Diagnostics")

    residuals_enh = model_enhanced.resid
    omni_stat_e, omni_p_e = omni_normtest(residuals_enh)
    jb_stat_e, jb_p_e, skew_e, kurtosis_e = jarque_bera(residuals_enh)
    dw_stat_e = durbin_watson(residuals_enh)

    diag_enh = pd.DataFrame({
        "Metric": [
            "Omnibus", "Prob(Omnibus)",
            "Jarque-Bera", "Prob(JB)",
            "Skew", "Kurtosis",
            "Durbin-Watson", "Condition Number"
        ],
        "Value": [
            omni_stat_e, omni_p_e,
            jb_stat_e, jb_p_e,
            skew_e, kurtosis_e,
            dw_stat_e, model_enhanced.condition_number
        ]
    })

    st.dataframe(diag_enh.round(3))

    # =========================
    # ENHANCED MODEL VIF
    # =========================
    st.subheader("Enhanced Model — VIF Check")

    vif_enh = pd.DataFrame({
        "Variable": X_enh.columns,
        "VIF": [variance_inflation_factor(X_enh.values, i) for i in range(X_enh.shape[1])]
    })
    vif_enh = vif_enh[vif_enh["Variable"] != "const"]
    vif_enh = vif_enh.sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_enh["Status"] = vif_enh["VIF"].apply(vif_flag)
    vif_enh["VIF"] = vif_enh["VIF"].round(2)

    st.dataframe(vif_enh)


    # ============================================================
    # 3-MODEL COMPARISON
    # ============================================================
    st.header("Model Comparison: Original → Fixed → Enhanced")

    comparison_3 = pd.DataFrame({
        "Metric": [
            "R²",
            "Adjusted R²",
            "F-Statistic",
            "Condition Number",
            "Durbin-Watson",
            "Observations"
        ],
        "Original Model": [
            round(model.rsquared, 4),
            round(model.rsquared_adj, 4),
            round(model.fvalue, 1),
            round(model.condition_number, 1),
            round(dw_stat, 3),
            int(model.nobs)
        ],
        "Fixed Model": [
            round(model_fixed.rsquared, 4),
            round(model_fixed.rsquared_adj, 4),
            round(model_fixed.fvalue, 1),
            round(model_fixed.condition_number, 1),
            round(dw_stat_f, 3),
            int(model_fixed.nobs)
        ],
        "Enhanced Model": [
            round(model_enhanced.rsquared, 4),
            round(model_enhanced.rsquared_adj, 4),
            round(model_enhanced.fvalue, 1),
            round(model_enhanced.condition_number, 1),
            round(dw_stat_e, 3),
            int(model_enhanced.nobs)
        ]
    })

    st.dataframe(comparison_3)

    # Improvement summary
    cond_enhanced = model_enhanced.condition_number
    imp_total = ((cond_original - cond_enhanced) / cond_original) * 100

    st.markdown(f"""
    **Condition Number progression:**
    - Original: **{cond_original:,.0f}**
    - Fixed (drop Age + center): **{cond_fixed:,.0f}**
    - Enhanced (+ controls): **{cond_enhanced:,.0f}** ({imp_total:.1f}% total reduction)

    **R² progression:**
    - Original: **{model.rsquared:.4f}**
    - Fixed: **{model_fixed.rsquared:.4f}**
    - Enhanced: **{model_enhanced.rsquared:.4f}**
    """)


    # ============================================================
    # CONSULTANT HYPOTHESIS TEST
    # ============================================================
    st.header("Consultant Hypothesis Test")

    st.markdown("""
    **Hypothesis:** The supervisor suspects that **consultants' high salaries** are biasing
    the results — making it look like certifications increase salary when it's really the
    consultant premium driving the effect.

    **Test:** Compare the `IsConsult` and `IsCertified` coefficients across models. If the
    consultant effect was confounded, the coefficient should change significantly after
    adding location, role, and other controls.
    """)

    # Get consultant and certification coefficients across models
    def get_coef_safe(m, var):
        if var in m.params.index:
            return m.params[var], m.pvalues[var]
        return None, None

    models_to_compare = {
        "Original Model": model,
        "Fixed Model": model_fixed,
        "Enhanced Model": model_enhanced
    }

    hypothesis_vars = ["IsConsult", "IsCertified", "IsMember", "IsFemale"]

    hyp_rows = []
    for var in hypothesis_vars:
        row = {"Variable": var}
        for name, m in models_to_compare.items():
            coef, pval = get_coef_safe(m, var)
            if coef is not None:
                row[f"{name} Coef"] = round(coef, 0)
                row[f"{name} p"] = round(pval, 4)
            else:
                row[f"{name} Coef"] = "N/A"
                row[f"{name} p"] = "N/A"
        hyp_rows.append(row)

    hyp_df = pd.DataFrame(hyp_rows)
    st.dataframe(hyp_df)

    # Interpretation
    consult_orig, consult_p_orig = get_coef_safe(model, "IsConsult")
    consult_enh, consult_p_enh = get_coef_safe(model_enhanced, "IsConsult")
    cert_orig, cert_p_orig = get_coef_safe(model, "IsCertified")
    cert_enh, cert_p_enh = get_coef_safe(model_enhanced, "IsCertified")

    st.subheader("Interpretation")

    if consult_orig is not None and consult_enh is not None:
        consult_change = ((consult_enh - consult_orig) / abs(consult_orig)) * 100
        st.markdown(f"""
    **Consultant Effect (`IsConsult`):**
    - Original model: **${consult_orig:,.0f}** (p={consult_p_orig:.4f})
    - Enhanced model: **${consult_enh:,.0f}** (p={consult_p_enh:.4f})
    - Change: **{consult_change:+.1f}%**
    """)

        if abs(consult_change) > 20:
            st.warning(f"⚠️ The consultant premium changed by {consult_change:+.1f}% after adding controls. "
                       "This suggests the original estimate was partially confounded by other factors.")
        else:
            st.info(f"The consultant premium is relatively stable ({consult_change:+.1f}% change). "
                    "Consultants genuinely earn more, independent of location, role, and other factors.")

    if cert_orig is not None and cert_enh is not None:
        cert_change = ((cert_enh - cert_orig) / abs(cert_orig)) * 100
        st.markdown(f"""
    **Certification Effect (`IsCertified`):**
    - Original model: **${cert_orig:,.0f}** (p={cert_p_orig:.4f})
    - Enhanced model: **${cert_enh:,.0f}** (p={cert_p_enh:.4f})
    - Change: **{cert_change:+.1f}%**
    """)

        if cert_p_enh is not None and cert_p_enh > 0.05:
            st.error("🔴 **Certification is NOT statistically significant** in the enhanced model. "
                     "The apparent certification premium may be explained by other factors "
                     "(location, role, company size, etc.).")
        elif cert_enh is not None and abs(cert_change) > 30:
            st.warning(f"⚠️ The certification effect changed by {cert_change:+.1f}% after adding controls. "
                       "The original estimate was partially confounded.")
        else:
            st.success("✅ Certification effect remains significant and stable after adding controls. "
                       "This strengthens the causal claim.")

