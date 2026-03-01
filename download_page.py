"""
download_page.py
================
Streamlit page that provides:
  1. PowerPoint presentation download
  2. Individual chart PNG downloads
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from generate_ppt import generate_presentation

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Download Center — Salary Analysis",
    page_icon="📥",
    layout="wide",
)

DATA_FILE = "salary_usd_cleaned.csv"

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["SalaryUSD"] = pd.to_numeric(df["Salary_USD"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["IsCertified"] = df["AACECertified"].astype(str).str.contains("Yes", case=False, na=False)
    df["IsMember"] = df["Member"].astype(str).str.contains("Yes", case=False, na=False)
    df["IsFemale"] = df["Sex"].astype(str).str.contains("Female", case=False, na=False)
    df["EmploymentStatus"] = df["EmploymentStatus"].astype(str).str.strip().str.lower()
    df["EmploymentStatus"] = df["EmploymentStatus"].replace({"employed full-time": "full-time"})
    return df


# ============================================================
# CHART GENERATORS (standalone PNGs — white background)
# ============================================================

def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def chart_salary_histogram(df):
    salaries = df["SalaryUSD"].dropna()
    salaries = salaries[salaries.between(5000, 500000)]
    bucket = 5000
    bins = np.arange(0, salaries.max() + bucket, bucket)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(salaries, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(salaries.mean(), color="red", linewidth=2, linestyle="--",
               label=f"Mean: ${salaries.mean():,.0f}")
    ax.axvline(salaries.median(), color="orange", linewidth=2, linestyle="-.",
               label=f"Median: ${salaries.median():,.0f}")

    ax.set_xlabel("Salary (USD)")
    ax.set_ylabel("Count")
    ax.set_title("Salary Distribution — All Respondents")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_membership(df):
    data = {
        "Members": df[df["IsMember"]]["SalaryUSD"].mean(),
        "Non-Members": df[~df["IsMember"]]["SalaryUSD"].mean(),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(data.keys(), data.values(), color=["#00BCD4", "#FF5722"], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 800,
                f"${b.get_height():,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_title("AACE Membership Effect on Salary")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(data.values()) * 1.15)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_certification(df):
    data = {
        "Certified": df[df["IsCertified"]]["SalaryUSD"].mean(),
        "Non-Certified": df[~df["IsCertified"]]["SalaryUSD"].mean(),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(data.keys(), data.values(), color=["#4CAF50", "#9E9E9E"], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 800,
                f"${b.get_height():,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_title("AACE Certification Effect on Salary")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(data.values()) * 1.15)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_gender_gap(df):
    men = df[~df["IsFemale"]]["SalaryUSD"].dropna()
    women = df[df["IsFemale"]]["SalaryUSD"].dropna()
    gap_pct = ((men.mean() - women.mean()) / men.mean()) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Men", "Women"], [men.mean(), women.mean()],
                  color=["#3A86FF", "#FF006E"], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 800,
                f"${b.get_height():,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_title(f"Gender Pay Gap ({gap_pct:.1f}% lower for women)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(men.mean(), women.mean()) * 1.15)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_gender_by_education(df):
    df_g = df.dropna(subset=["SalaryUSD", "LevelOfEducation"]).copy()
    df_g["LevelOfEducation"] = df_g["LevelOfEducation"].str.strip().str.lower()
    df_g["LevelOfEducation"] = df_g["LevelOfEducation"].replace({
        "undergraduate/bachelor\u2019s degree": "bachelors",
        "undergraduate/bachelor's degree": "bachelors",
        "undergraduate or bachelor's degree": "bachelors",
        "graduate/master\u2019s degree": "masters",
        "graduate/master's degree": "masters",
        "graduate/doctoral degree": "doctoral",
        "graduate - masters degree": "masters",
        "graduate - doctoral degree": "doctoral",
        "undergraduate or bachelors degree": "bachelors",
        "associate degree": "associate",
        "high school": "high school",
    })

    edu_order = ["high school", "associate", "bachelors", "masters", "doctoral"]
    labels = ["High School", "Associate", "Bachelor's", "Master's", "Doctoral"]

    gender_edu = df_g.groupby(["LevelOfEducation", "IsFemale"])["SalaryUSD"].mean().unstack()
    gender_edu.columns = ["Men", "Women"]
    gender_edu = gender_edu.reindex([e for e in edu_order if e in gender_edu.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(gender_edu))
    w = 0.35
    ax.bar(x - w/2, gender_edu["Men"], w, color="#3A86FF", label="Men")
    ax.bar(x + w/2, gender_edu["Women"], w, color="#FF006E", label="Women")

    disp = [labels[edu_order.index(e)] if e in edu_order else e.title()
            for e in gender_edu.index]
    ax.set_xticks(x)
    ax.set_xticklabels(disp)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_title("Gender Pay Gap by Education Level")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, (m, wv) in enumerate(zip(gender_edu["Men"], gender_edu["Women"])):
        ax.text(i - 0.175, m + 800, f"${m:,.0f}", ha="center", fontsize=7)
        ax.text(i + 0.175, wv + 800, f"${wv:,.0f}", ha="center", fontsize=7)

    plt.tight_layout()
    return _fig_to_png(fig)


def chart_satisfaction(df):
    df_s = df.dropna(subset=["JobSatisfaction", "SalaryUSD"]).copy()
    df_s["JobSatisfaction"] = df_s["JobSatisfaction"].str.strip().str.lower()
    order = ["very dissatisfied", "somewhat dissatisfied",
             "somewhat satisfied", "very satisfied"]
    labels = ["Very\nDissatisfied", "Somewhat\nDissatisfied",
              "Somewhat\nSatisfied", "Very\nSatisfied"]

    dist = df_s["JobSatisfaction"].value_counts(normalize=True) * 100
    dist = dist.reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_bar = ["#F44336", "#FF9800", "#8BC34A", "#4CAF50"]
    bars = ax.bar(range(len(dist)), dist.values, color=colors_bar, width=0.6)
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(labels)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of Respondents")
    ax.set_title("Job Satisfaction Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_industry_gap(df):
    df_ind = df.dropna(subset=["SalaryUSD", "Industry"]).copy()
    df_ind["Industry"] = df_ind["Industry"].str.strip().str.lower()
    top = df_ind["Industry"].value_counts().head(6).index
    df_ind = df_ind[df_ind["Industry"].isin(top)]

    gender_ind = df_ind.groupby(["Industry", "IsFemale"])["SalaryUSD"].mean().unstack()
    gender_ind.columns = ["Men", "Women"]
    gender_ind["Gap %"] = ((gender_ind["Men"] - gender_ind["Women"]) / gender_ind["Men"]) * 100
    gender_ind = gender_ind.sort_values("Gap %", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(gender_ind.index.str.title(), gender_ind["Gap %"], color="#FF006E", height=0.5)
    for i, v in enumerate(gender_ind["Gap %"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Gender Pay Gap (%)")
    ax.set_title("Gender Pay Gap by Industry")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_consulting(df):
    df_c = df.dropna(subset=["SalaryUSD", "Consult"]).copy()
    df_c["IsConsultant"] = df_c["Consult"].astype(str).str.contains("Yes", case=False, na=False)
    data = {
        "Consultant": df_c[df_c["IsConsultant"]]["SalaryUSD"].mean(),
        "Non-Consultant": df_c[~df_c["IsConsultant"]]["SalaryUSD"].mean(),
    }
    premium = data["Consultant"] - data["Non-Consultant"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(data.keys(), data.values(), color=["#FFC107", "#607D8B"], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 800,
                f"${b.get_height():,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_title(f"Consultant Premium (+${premium:,.0f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(data.values()) * 1.15)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_ols_impact(df):
    import statsmodels.api as sm

    df_reg = df.dropna(subset=["SalaryUSD", "YearsOfExperience", "Age"]).copy()
    df_reg["IsManager"] = df_reg["ManagerialDuties"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_reg["IsConsult"] = df_reg["Consult"].astype(str).str.contains("Yes", case=False, na=False).astype(int)
    df_reg["IsCertified"] = df_reg["IsCertified"].astype(int)
    df_reg["IsMember"] = df_reg["IsMember"].astype(int)
    df_reg["IsFemale"] = df_reg["IsFemale"].astype(int)

    X = df_reg[["YearsOfExperience", "IsCertified", "IsMember",
                "IsFemale", "IsManager", "IsConsult"]].astype(float)
    X = sm.add_constant(X)
    y = df_reg["SalaryUSD"].astype(float)
    model = sm.OLS(y, X).fit()

    coefs = model.params.drop("const")
    pvals = model.pvalues.drop("const")
    coef_df = pd.DataFrame({"Coefficient": coefs, "p": pvals})
    coef_df["AbsImpact"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsImpact", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if p < 0.05 else "#9E9E9E" for p in coef_df["p"]]
    ax.barh(coef_df.index, coef_df["Coefficient"], color=colors, height=0.5)
    for i, (v, p) in enumerate(zip(coef_df["Coefficient"], coef_df["p"])):
        label = f"${v:,.0f}" + (" *" if p < 0.05 else "")
        ax.text(v, i, label, va="center", fontsize=9)

    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Impact on Salary (USD)")
    ax.set_title(f"OLS Key Impact Factors (R² = {model.rsquared:.3f})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _fig_to_png(fig)


def chart_year_comparison(df):
    df_2015 = df[df["SurveyYear"] == 2015]["SalaryUSD"].dropna()
    df_2023 = df[df["SurveyYear"] == 2023]["SalaryUSD"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = {
        "Mean": [df_2015.mean(), df_2023.mean()],
        "Median": [df_2015.median(), df_2023.median()],
    }
    x = np.arange(2)
    w = 0.3
    bars1 = ax.bar(x - w/2, data["Mean"], w, color="#00BCD4", label="Mean")
    bars2 = ax.bar(x + w/2, data["Median"], w, color="#FFC107", label="Median")
    ax.set_xticks(x)
    ax.set_xticklabels(["2015", "2023"])
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 800,
                f"${b.get_height():,.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Salary Comparison: 2015 vs 2023")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _fig_to_png(fig)


# ============================================================
# CHART REGISTRY
# ============================================================
CHARTS = [
    ("Salary Distribution — All Respondents", "salary_distribution.png", chart_salary_histogram),
    ("2015 vs 2023 Comparison", "year_comparison.png", chart_year_comparison),
    ("AACE Membership Effect", "membership_effect.png", chart_membership),
    ("AACE Certification Effect", "certification_effect.png", chart_certification),
    ("Gender Pay Gap — Overall", "gender_gap_overall.png", chart_gender_gap),
    ("Gender Pay Gap by Education", "gender_gap_education.png", chart_gender_by_education),
    ("Gender Pay Gap by Industry", "gender_gap_industry.png", chart_industry_gap),
    ("Job Satisfaction Distribution", "satisfaction_distribution.png", chart_satisfaction),
    ("Consulting Premium", "consulting_premium.png", chart_consulting),
    ("OLS Key Impact Factors", "ols_impact_factors.png", chart_ols_impact),
]


# ============================================================
# PAGE UI
# ============================================================

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0B1D3A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #607D8B;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #009688;
        border-bottom: 2px solid #009688;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .chart-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Download Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Download the PowerPoint presentation and individual analysis charts</div>', unsafe_allow_html=True)

df = load_data()

# ============================================================
# SECTION 1: POWERPOINT DOWNLOAD
# ============================================================
st.markdown('<div class="section-title">PowerPoint Presentation</div>', unsafe_allow_html=True)

col_info, col_btn = st.columns([3, 1])

with col_info:
    st.markdown("""
    Generate a **19-slide story-driven presentation** covering:
    - Executive summary & data overview
    - Salary distribution analysis
    - Membership & certification effects
    - Gender pay gap (overall, by education, by industry)
    - Job satisfaction insights
    - OLS regression & model evolution
    - Consultant hypothesis test
    - Key takeaways & recommendations
    """)

with col_btn:
    st.markdown("")
    st.markdown("")
    if st.button("Generate PowerPoint", type="primary", use_container_width=True):
        with st.spinner("Building presentation... this may take a moment"):
            ppt_bytes = generate_presentation(DATA_FILE)
            st.session_state["ppt_bytes"] = ppt_bytes.getvalue()

    if "ppt_bytes" in st.session_state:
        st.download_button(
            label="Download .pptx",
            data=st.session_state["ppt_bytes"],
            file_name="Salary_Analysis_Presentation.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )
        st.success("Presentation ready!")

st.divider()

# ============================================================
# SECTION 2: INDIVIDUAL CHART DOWNLOADS
# ============================================================
st.markdown('<div class="section-title">Individual Chart Downloads (PNG)</div>', unsafe_allow_html=True)

st.markdown("Preview and download any chart used in the analysis. All charts are exported as high-resolution PNG images.")

# Initialize chart cache
if "chart_cache" not in st.session_state:
    st.session_state["chart_cache"] = {}

# Generate all charts button
if st.button("Generate All Charts", type="secondary"):
    with st.spinner("Generating charts..."):
        for title, filename, func in CHARTS:
            st.session_state["chart_cache"][filename] = func(df)
    st.success(f"All {len(CHARTS)} charts generated!")

st.markdown("")

# Display chart grid
for i in range(0, len(CHARTS), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= len(CHARTS):
            break
        title, filename, func = CHARTS[idx]
        with col:
            st.markdown(f"**{title}**")

            # Generate on demand if not cached
            if filename not in st.session_state["chart_cache"]:
                if st.button(f"Generate", key=f"gen_{filename}"):
                    with st.spinner("Generating..."):
                        st.session_state["chart_cache"][filename] = func(df)
                    st.rerun()
            else:
                st.image(st.session_state["chart_cache"][filename], use_column_width=True)
                st.download_button(
                    label=f"Download {filename}",
                    data=st.session_state["chart_cache"][filename],
                    file_name=filename,
                    mime="image/png",
                    key=f"dl_{filename}",
                )

            st.markdown("---")

# ============================================================
# FOOTER
# ============================================================
st.markdown("")
st.markdown(
    "<div style='text-align:center; color:#9E9E9E; font-size:0.85rem;'>"
    "AACE Salary Survey Analysis | Download Center"
    "</div>",
    unsafe_allow_html=True,
)
