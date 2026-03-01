import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    df["SalaryUSD"] = pd.to_numeric(df["Salary_USD"], errors="coerce")
    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df["IsFemale"] = df["Sex"].astype(str).str.contains("Female", case=False, na=False)

    # Clean employment status
    df["EmploymentStatus"] = df["EmploymentStatus"].astype(str).str.strip().str.lower()
    df["EmploymentStatus"] = df["EmploymentStatus"].replace({
        "employed full-time": "full-time"
    })

    return df


# =========================
# HISTOGRAM PLOT FUNCTION
# =========================
def plot_histogram(df_input, title, color="steelblue", bucket_size=5000):
    """
    Plot a salary histogram with $5,000 buckets.
    Includes mean, median, std, skewness, and kurtosis annotations.
    """
    df_plot = df_input.dropna(subset=["SalaryUSD"]).copy()
    df_plot = df_plot[df_plot["SalaryUSD"].between(5000, 500000)]

    if len(df_plot) < 10:
        st.warning(f"Not enough data for: {title}")
        return

    salaries = df_plot["SalaryUSD"]

    # Statistics
    mean_sal = salaries.mean()
    median_sal = salaries.median()
    std_sal = salaries.std()
    skewness = stats.skew(salaries, nan_policy="omit")
    kurt = stats.kurtosis(salaries, nan_policy="omit")  # excess kurtosis
    n = len(salaries)

    # Mode: midpoint of the most frequent bin
    bin_min_pre = int(np.floor(salaries.min() / bucket_size) * bucket_size)
    bin_max_pre = int(np.ceil(salaries.max() / bucket_size) * bucket_size) + bucket_size
    bins_pre = np.arange(bin_min_pre, bin_max_pre, bucket_size)
    counts_pre, edges_pre = np.histogram(salaries, bins=bins_pre)
    mode_bin_idx = np.argmax(counts_pre)
    mode_sal = (edges_pre[mode_bin_idx] + edges_pre[mode_bin_idx + 1]) / 2

    # Bin edges: $5,000 buckets
    bin_min = int(np.floor(salaries.min() / bucket_size) * bucket_size)
    bin_max = int(np.ceil(salaries.max() / bucket_size) * bucket_size) + bucket_size
    bins = np.arange(bin_min, bin_max, bucket_size)

    # -------------------------
    # PLOT
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    counts, edges, patches = ax.hist(
        salaries,
        bins=bins,
        color=color,
        edgecolor="white",
        alpha=0.85,
        linewidth=0.6
    )

    # Mean line
    ax.axvline(
        mean_sal,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Mean: ${mean_sal:,.0f}"
    )

    # Median line
    ax.axvline(
        median_sal,
        color="orange",
        linewidth=2,
        linestyle="-.",
        label=f"Median: ${median_sal:,.0f}"
    )

    # Mode line
    ax.axvline(
        mode_sal,
        color="green",
        linewidth=2,
        linestyle=":",
        label=f"Mode: ${mode_sal:,.0f}"
    )

    # ±1σ shading
    ax.axvspan(
        mean_sal - std_sal,
        mean_sal + std_sal,
        alpha=0.08,
        color="red",
        label=f"±1σ (${mean_sal - std_sal:,.0f} – ${mean_sal + std_sal:,.0f})"
    )

    ax.set_xlabel("Salary (USD)", fontsize=11)
    ax.set_ylabel("Number of Respondents", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # X-axis formatting: show $K labels
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")
    )

    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # STATISTICS TABLE
    # -------------------------
    stats_data = {
        "Metric": [
            "N (Count)",
            "Mean Salary",
            "Median Salary",
            "Mode Salary (bin midpoint)",
            "Std Deviation",
            "Skewness",
            "Excess Kurtosis",
            "Min Salary",
            "Max Salary",
            "25th Percentile (Q1)",
            "75th Percentile (Q3)",
            "IQR"
        ],
        "Value": [
            f"{n:,}",
            f"${mean_sal:,.0f}",
            f"${median_sal:,.0f}",
            f"${mode_sal:,.0f}",
            f"${std_sal:,.0f}",
            f"{skewness:.4f}",
            f"{kurt:.4f}",
            f"${salaries.min():,.0f}",
            f"${salaries.max():,.0f}",
            f"${salaries.quantile(0.25):,.0f}",
            f"${salaries.quantile(0.75):,.0f}",
            f"${salaries.quantile(0.75) - salaries.quantile(0.25):,.0f}"
        ]
    }

    st.dataframe(pd.DataFrame(stats_data), hide_index=True)

    # -------------------------
    # INTERPRETATION
    # -------------------------
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


# =========================
# APP
# =========================
st.set_page_config(page_title="Salary Histogram Analysis", layout="wide")
st.title("📊 Salary Distribution – Histogram Analysis")
st.markdown("Salary histograms in **$5,000 buckets** to visualize the shape, spread, and kurtosis of the dataset.")

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

# Survey Year Filter
year_options = sorted(df["SurveyYear"].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Survey Year",
    year_options,
    default=year_options
)

# Employment Status Filter
employment_options = sorted(df["EmploymentStatus"].dropna().unique())
selected_employment = st.sidebar.multiselect(
    "Employment Status",
    employment_options,
    default=employment_options
)

# Salary Range Filter
salary_min = int(df["SalaryUSD"].dropna().min())
salary_max = int(df["SalaryUSD"].dropna().max())

salary_range = st.sidebar.slider(
    "Salary Range (USD)",
    min_value=0,
    max_value=min(salary_max, 500000),
    value=(5000, 300000),
    step=5000
)

# Bucket Size
bucket_size = st.sidebar.selectbox(
    "Bucket Size (USD)",
    [2500, 5000, 10000, 15000, 20000, 25000],
    index=1  # default $5,000
)

# Apply Filters
df_filtered = df[
    (df["SurveyYear"].isin(selected_years)) &
    (df["EmploymentStatus"].isin(selected_employment)) &
    (df["SalaryUSD"].between(salary_range[0], salary_range[1]))
]

st.write(f"**Total filtered records:** {len(df_filtered):,}")

# =========================
# SECTION 1: ALL RESPONDENTS
# =========================
st.header("All Respondents")

plot_histogram(
    df_filtered,
    "Salary Distribution – All Respondents",
    color="steelblue",
    bucket_size=bucket_size
)

# =========================
# SECTION 2: MEN vs WOMEN (SIDE-BY-SIDE)
# =========================
st.header("Men vs Women")

df_men = df_filtered[~df_filtered["IsFemale"]]
df_women = df_filtered[df_filtered["IsFemale"]]

c1, c2 = st.columns(2)

with c1:
    plot_histogram(
        df_men,
        "Salary Distribution – Men",
        color="#3A86FF",
        bucket_size=bucket_size
    )

with c2:
    plot_histogram(
        df_women,
        "Salary Distribution – Women",
        color="#FF006E",
        bucket_size=bucket_size
    )

# =========================
# SECTION 3: 2015 vs 2023 SPLIT
# =========================
if len(selected_years) > 1 and 2015 in selected_years and 2023 in selected_years:

    st.header("Year-over-Year Comparison (2015 vs 2023)")

    # --- All respondents by year ---
    st.subheader("All Respondents")

    c1, c2 = st.columns(2)

    with c1:
        plot_histogram(
            df_filtered[df_filtered["SurveyYear"] == 2015],
            "All Respondents – 2015",
            color="#457B9D",
            bucket_size=bucket_size
        )

    with c2:
        plot_histogram(
            df_filtered[df_filtered["SurveyYear"] == 2023],
            "All Respondents – 2023",
            color="#E63946",
            bucket_size=bucket_size
        )

    # --- Men by year ---
    st.subheader("Men")

    c1, c2 = st.columns(2)

    with c1:
        plot_histogram(
            df_men[df_men["SurveyYear"] == 2015],
            "Men – 2015",
            color="#264653",
            bucket_size=bucket_size
        )

    with c2:
        plot_histogram(
            df_men[df_men["SurveyYear"] == 2023],
            "Men – 2023",
            color="#2A9D8F",
            bucket_size=bucket_size
        )

    # --- Women by year ---
    st.subheader("Women")

    c1, c2 = st.columns(2)

    with c1:
        plot_histogram(
            df_women[df_women["SurveyYear"] == 2015],
            "Women – 2015",
            color="#F4A261",
            bucket_size=bucket_size
        )

    with c2:
        plot_histogram(
            df_women[df_women["SurveyYear"] == 2023],
            "Women – 2023",
            color="#E76F51",
            bucket_size=bucket_size
        )


# =========================
# SECTION 4: OVERLAY COMPARISON
# =========================
st.header("Overlay: Men vs Women")

df_men_sal = df_men["SalaryUSD"].dropna()
df_women_sal = df_women["SalaryUSD"].dropna()

df_men_sal = df_men_sal[df_men_sal.between(5000, 500000)]
df_women_sal = df_women_sal[df_women_sal.between(5000, 500000)]

if len(df_men_sal) > 10 and len(df_women_sal) > 10:

    all_salaries = pd.concat([df_men_sal, df_women_sal])
    bin_min = int(np.floor(all_salaries.min() / bucket_size) * bucket_size)
    bin_max = int(np.ceil(all_salaries.max() / bucket_size) * bucket_size) + bucket_size
    bins = np.arange(bin_min, bin_max, bucket_size)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.hist(
        df_men_sal,
        bins=bins,
        alpha=0.55,
        color="#3A86FF",
        edgecolor="white",
        linewidth=0.5,
        label=f"Men (n={len(df_men_sal):,})"
    )

    ax.hist(
        df_women_sal,
        bins=bins,
        alpha=0.55,
        color="#FF006E",
        edgecolor="white",
        linewidth=0.5,
        label=f"Women (n={len(df_women_sal):,})"
    )

    # Compute mode (bin midpoint) for each gender
    men_counts, men_edges = np.histogram(df_men_sal, bins=bins)
    men_mode_idx = np.argmax(men_counts)
    men_mode = (men_edges[men_mode_idx] + men_edges[men_mode_idx + 1]) / 2

    women_counts, women_edges = np.histogram(df_women_sal, bins=bins)
    women_mode_idx = np.argmax(women_counts)
    women_mode = (women_edges[women_mode_idx] + women_edges[women_mode_idx + 1]) / 2

    # Mean lines
    ax.axvline(df_men_sal.mean(), color="#3A86FF", linewidth=2, linestyle="--",
               label=f"Men Mean: ${df_men_sal.mean():,.0f}")
    ax.axvline(df_women_sal.mean(), color="#FF006E", linewidth=2, linestyle="--",
               label=f"Women Mean: ${df_women_sal.mean():,.0f}")

    # Median lines
    ax.axvline(df_men_sal.median(), color="#3A86FF", linewidth=2, linestyle="-.",
               label=f"Men Median: ${df_men_sal.median():,.0f}")
    ax.axvline(df_women_sal.median(), color="#FF006E", linewidth=2, linestyle="-.",
               label=f"Women Median: ${df_women_sal.median():,.0f}")

    # Mode lines
    ax.axvline(men_mode, color="#3A86FF", linewidth=2, linestyle=":",
               label=f"Men Mode: ${men_mode:,.0f}")
    ax.axvline(women_mode, color="#FF006E", linewidth=2, linestyle=":",
               label=f"Women Mode: ${women_mode:,.0f}")

    ax.set_xlabel("Salary (USD)", fontsize=11)
    ax.set_ylabel("Number of Respondents", fontsize=11)
    ax.set_title("Overlaid Salary Distribution – Men vs Women", fontsize=13, fontweight="bold")

    ax.ticklabel_format(style="plain", axis="x")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary comparison table
    gap = df_men_sal.mean() - df_women_sal.mean()
    gap_pct = (gap / df_men_sal.mean()) * 100

    summary = pd.DataFrame({
        "Metric": ["Count", "Mean", "Median", "Mode (bin midpoint)", "Std Dev", "Skewness", "Excess Kurtosis"],
        "Men": [
            f"{len(df_men_sal):,}",
            f"${df_men_sal.mean():,.0f}",
            f"${df_men_sal.median():,.0f}",
            f"${men_mode:,.0f}",
            f"${df_men_sal.std():,.0f}",
            f"{stats.skew(df_men_sal):.4f}",
            f"{stats.kurtosis(df_men_sal):.4f}"
        ],
        "Women": [
            f"{len(df_women_sal):,}",
            f"${df_women_sal.mean():,.0f}",
            f"${df_women_sal.median():,.0f}",
            f"${women_mode:,.0f}",
            f"${df_women_sal.std():,.0f}",
            f"{stats.skew(df_women_sal):.4f}",
            f"{stats.kurtosis(df_women_sal):.4f}"
        ]
    })

    st.dataframe(summary, hide_index=True)

    st.markdown(f"""
**Gender Pay Gap Summary:**
- Mean salary gap: **${gap:,.0f}** ({gap_pct:.1f}% lower for women)
- Men mean: ${df_men_sal.mean():,.0f} | Women mean: ${df_women_sal.mean():,.0f}
""")

else:
    st.warning("Not enough data for Men or Women to create the overlay chart.")
