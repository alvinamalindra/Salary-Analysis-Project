import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Salary vs Experience — SPC Best Fit Mode")

# Load data
uploaded_file = st.file_uploader("Upload Salary Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Required columns
    # YearsOfExperience, SalaryUSD, MemberStatus

    members_only = st.checkbox("Members Only", value=True)

    if members_only:
        df = df[df["MemberStatus"] == "Member"]

    # Compute BASELINE stats (never changes)
    base_mean = df["SalaryUSD"].mean()
    base_std = df["SalaryUSD"].std()

    UCL = base_mean + 3 * base_std
    LCL = base_mean - 3 * base_std

    remove_outliers = st.checkbox("Remove +3σ Outliers", value=False)

    # Filter data ONLY — DO NOT recompute sigma
    plot_df = df.copy()
    if remove_outliers:
        plot_df = plot_df[plot_df["SalaryUSD"] <= UCL]

    x = plot_df["YearsOfExperience"]
    y = plot_df["SalaryUSD"]

    # BEST FIT (Polynomial curve)
    if len(x) > 3:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = p(x_fit)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter points
    ax.scatter(x, y, alpha=0.35)

    # Best Fit Line
    if len(x) > 3:
        ax.plot(x_fit, y_fit, linewidth=2)

    # Mean & Control Limits (baseline)
    ax.axhline(base_mean, linewidth=2)
    ax.axhline(UCL, linestyle="--")
    ax.axhline(LCL, linestyle="--")

    # Axis scaling includes ORIGINAL UCL/LCL
    ymin = min(y.min(), LCL) * 0.95
    ymax = max(y.max(), UCL) * 1.05
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Members Salary vs Experience")

    # ---- LABELS ON RIGHT SIDE ----
    right_x = x.max() * 1.02

    ax.text(right_x, base_mean, "Mean", va="center")
    ax.text(right_x, UCL, "UCL (+3σ)", va="center")
    ax.text(right_x, LCL, "LCL (−3σ)", va="center")

    # Remove legend clutter
    ax.legend([], [], frameon=False)

    st.pyplot(fig)

    # Stats summary
    st.markdown("### Baseline SPC Statistics (Fixed)")
    st.write(f"Mean: ${base_mean:,.0f}")
    st.write(f"Std Dev: ${base_std:,.0f}")
    st.write(f"UCL (+3σ): ${UCL:,.0f}")
    st.write(f"LCL (−3σ): ${LCL:,.0f}")
