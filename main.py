"""
============================================================
E-Commerce Conversion Rate Optimization (CRO) Analysis
Author  : [RAY]
Purpose : Philips Business Analyst Portfolio — GitHub Showcase
Dataset : Cosmetics E-Commerce User Behaviour (Oct 2019)
============================================================
Sections
--------
1. Configuration & Imports
2. Data Loading & Cleaning
3. Funnel Analysis (View → Cart → Purchase)
4. Brand Revenue Analysis (Top 10)
5. Visualisation — Conversion Funnel
6. Visualisation — Top 10 Brand Revenue
7. Actionable Business Insights (Console Output)
"""

# =============================================================
# 1. CONFIGURATION & IMPORTS
# =============================================================
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# ---------- Philips-inspired enterprise colour palette ----------
PHILIPS_BLUE      = "#0B5ED7"   # primary brand blue
PHILIPS_DEEP      = "#003087"   # deep navy — headlines / axes
PHILIPS_LIGHT     = "#4A9FFF"   # mid-sky — secondary bars
PHILIPS_ACCENT    = "#00C2FF"   # electric cyan — highlights
PHILIPS_BG        = "#F0F5FF"   # page background tint
PHILIPS_GRID      = "#D6E4FF"   # subtle grid lines
TEXT_PRIMARY      = "#0A1628"   # near-black body text
TEXT_SECONDARY    = "#4A5568"   # medium-grey subtitles
GRADIENT_COLORS   = [           # funnel stage gradient
    "#003087", "#0B5ED7", "#4A9FFF"
]

# ---------- Output directory ----------
VISUALS_DIR = "visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

# ---------- Global matplotlib theme ----------
plt.rcParams.update({
    "figure.facecolor":  PHILIPS_BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    PHILIPS_GRID,
    "axes.labelcolor":   TEXT_PRIMARY,
    "axes.titlecolor":   PHILIPS_DEEP,
    "axes.grid":         True,
    "grid.color":        PHILIPS_GRID,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "grid.alpha":        0.8,
    "xtick.color":       TEXT_SECONDARY,
    "ytick.color":       TEXT_SECONDARY,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
})


# =============================================================
# 2. DATA LOADING & CLEANING
# =============================================================
def load_and_clean(filepath: str, nrows: int = 1_000_000) -> pd.DataFrame:
    """
    Load the raw CSV and apply business-rule cleaning.

    Parameters
    ----------
    filepath : str
        Path to the source CSV file.
    nrows : int
        Maximum rows to read — guards against memory overflow on
        large monthly datasets (default: 1 000 000).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for analysis.
    """
    print("=" * 60)
    print("  PHILIPS CRO ANALYSIS — Data Loading")
    print("=" * 60)

    df = pd.read_csv(
        filepath,
        nrows=nrows,
        parse_dates=["event_time"],
        dtype={
            "event_type":  "category",
            "product_id":  "int32",
            "brand":       "str",
            "user_id":     "int32",
            "user_session": "str",
        },
        low_memory=True,
    )

    raw_rows = len(df)
    print(f"  ✔ Rows loaded        : {raw_rows:,}")

    # --- Business rule: remove zero / negative price records ---
    df = df[df["price"] > 0].copy()
    print(f"  ✔ Rows after cleaning: {len(df):,}  "
          f"(removed {raw_rows - len(df):,} anomalous records)")
    print(f"  ✔ Date range         : "
          f"{df['event_time'].min().date()} → {df['event_time'].max().date()}")
    print(f"  ✔ Unique sessions    : {df['user_session'].nunique():,}")
    print()

    return df


# =============================================================
# 3. FUNNEL ANALYSIS
# =============================================================
def compute_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the View → Cart → Purchase conversion funnel.

    Each stage counts *unique user sessions* — this prevents
    inflated numbers from repeat page-view events and aligns
    with industry-standard funnel methodology.

    Returns
    -------
    pd.DataFrame
        Columns: stage, sessions, conv_from_prev, conv_from_top
    """
    stages = ["view", "cart", "purchase"]
    labels = ["View", "Cart", "Purchase"]

    session_counts = [
        df.loc[df["event_type"] == s, "user_session"].nunique()
        for s in stages
    ]

    funnel = pd.DataFrame({
        "stage":    labels,
        "sessions": session_counts,
    })

    # Conversion rate vs. previous stage (step-over-step)
    funnel["conv_from_prev"] = funnel["sessions"] / funnel["sessions"].shift(1) * 100
    funnel.loc[0, "conv_from_prev"] = 100.0   # top of funnel = 100 %

    # Conversion rate vs. top of funnel (overall)
    funnel["conv_from_top"] = funnel["sessions"] / funnel["sessions"].iloc[0] * 100

    return funnel


# =============================================================
# 4. BRAND REVENUE ANALYSIS
# =============================================================
def compute_top_brands(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identify the top revenue-generating brands from purchase events.

    Revenue = Σ(price) for all purchase records per brand.
    Brands with missing/null names are excluded to maintain
    data quality in stakeholder-facing outputs.

    Parameters
    ----------
    df    : cleaned dataframe
    top_n : number of brands to return (default 10)

    Returns
    -------
    pd.DataFrame  — columns: brand, revenue, pct_of_total
    """
    purchases = df[df["event_type"] == "purchase"].copy()
    purchases = purchases[purchases["brand"].notna() & (purchases["brand"] != "nan")]

    brand_revenue = (
        purchases.groupby("brand", observed=True)["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "revenue"})
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    total_revenue = brand_revenue["revenue"].sum()
    brand_revenue["pct_of_total"] = brand_revenue["revenue"] / total_revenue * 100

    return brand_revenue, purchases["price"].sum()


# =============================================================
# 5. VISUALISATION — CONVERSION FUNNEL
# =============================================================
def plot_conversion_funnel(funnel: pd.DataFrame, save_path: str) -> None:
    """
    Render an enterprise-grade conversion funnel bar chart.

    Design language: Philips signature deep-blue gradient,
    clean sans-serif typography, and explicit data labels
    that surface both absolute session counts and step-over-step
    conversion percentages for immediate executive readability.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(PHILIPS_BG)
    ax.set_facecolor("white")

    stages   = funnel["stage"].tolist()
    sessions = funnel["sessions"].tolist()
    n        = len(stages)

    # --- Gradient bars (deep → light) ---
    bars = ax.bar(
        stages,
        sessions,
        color=GRADIENT_COLORS[:n],
        width=0.52,
        zorder=3,
        edgecolor="white",
        linewidth=1.4,
    )

    # --- Connector arrows between bars ---
    for i in range(n - 1):
        x_mid = i + 0.5
        y_top = max(sessions[i], sessions[i + 1]) * 0.5
        ax.annotate(
            "",
            xy=(i + 1 - 0.27, sessions[i + 1] * 0.85),
            xytext=(i + 0.27, sessions[i] * 0.85),
            arrowprops=dict(
                arrowstyle="-|>",
                color=PHILIPS_ACCENT,
                lw=1.8,
            ),
            zorder=4,
        )

    # --- Data labels: session count (top) + conversion % (inside bar) ---
    for idx, (bar, row) in enumerate(zip(bars, funnel.itertuples())):
        height = bar.get_height()
        x      = bar.get_x() + bar.get_width() / 2

        # Session count above bar
        ax.text(
            x, height + max(sessions) * 0.015,
            f"{height:,.0f}",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color=PHILIPS_DEEP,
        )

        # Step conversion % inside bar (skip top-of-funnel 100 %)
        if idx > 0:
            label = f"↓ {row.conv_from_prev:.1f}% from prev."
            ax.text(
                x, height / 2,
                label,
                ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                x, height / 2,
                "Top of Funnel",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color="white",
            )

    # --- Overall conversion badge (View → Purchase) ---
    overall = funnel.loc[funnel["stage"] == "Purchase", "conv_from_top"].values[0]
    ax.text(
        0.98, 0.97,
        f"Overall Conversion\n{overall:.2f}%",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=PHILIPS_BLUE,
            edgecolor=PHILIPS_ACCENT,
            linewidth=1.5,
        ),
    )

    # --- Axes & labels ---
    ax.set_title(
        "Customer Journey Conversion Funnel\nView  →  Cart  →  Purchase",
        pad=18, color=PHILIPS_DEEP,
    )
    ax.set_ylabel("Unique User Sessions", labelpad=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Funnel Stage", labelpad=8, color=TEXT_PRIMARY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, max(sessions) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Footer watermark ---
    fig.text(
        0.99, 0.005,
        "Source: Cosmetics E-Commerce Dataset | Philips BA Portfolio",
        ha="right", fontsize=8, color=TEXT_SECONDARY, style="italic",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved → {save_path}")


# =============================================================
# 6. VISUALISATION — TOP 10 BRAND REVENUE
# =============================================================
def plot_top_brands(brand_df: pd.DataFrame, total_revenue: float,
                    save_path: str) -> None:
    """
    Render a horizontal bar chart for Top 10 brand revenue contribution.

    The chart uses a sequential blue palette that reinforces brand
    hierarchy at a glance. Dual data labels (absolute £ revenue +
    percentage share) give finance stakeholders all the numbers
    they need without consulting raw tables.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(PHILIPS_BG)
    ax.set_facecolor("white")

    # Reverse so highest bar is at top
    plot_df = brand_df.iloc[::-1].reset_index(drop=True)

    # Sequential palette from mid-blue → deep-navy
    palette = sns.color_palette(
        "Blues_r", n_colors=len(plot_df) + 3
    )[2: len(plot_df) + 2]

    bars = ax.barh(
        plot_df["brand"],
        plot_df["revenue"],
        color=palette,
        height=0.62,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # --- Data labels ---
    max_rev = plot_df["revenue"].max()
    for bar, row in zip(bars, plot_df.itertuples()):
        width = bar.get_width()

        # Revenue value (just inside bar-end for long bars)
        label_x = width - max_rev * 0.01 if width > max_rev * 0.25 else width + max_rev * 0.01
        ha       = "right"                if width > max_rev * 0.25 else "left"
        color    = "white"                if width > max_rev * 0.25 else PHILIPS_DEEP

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"${width:,.0f}",
            va="center", ha=ha,
            fontsize=10.5, fontweight="bold",
            color=color,
        )

        # Percentage badge on the right margin
        ax.text(
            max_rev * 1.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row.pct_of_total:.1f}%",
            va="center", ha="left",
            fontsize=9.5, color=TEXT_SECONDARY,
        )

    # Column header for pct column
    ax.text(
        max_rev * 1.015, len(plot_df) - 0.05,
        "Share",
        va="bottom", ha="left",
        fontsize=9, color=TEXT_SECONDARY,
        fontweight="bold",
    )

    # Vertical reference line at mean revenue
    mean_rev = plot_df["revenue"].mean()
    ax.axvline(mean_rev, color=PHILIPS_ACCENT, linestyle="--",
               linewidth=1.2, alpha=0.8, zorder=2)
    ax.text(
        mean_rev, -0.75,
        f"Avg: ${mean_rev:,.0f}",
        ha="center", va="top",
        fontsize=8.5, color=PHILIPS_ACCENT,
    )

    # --- Axes & styling ---
    ax.set_title(
        "Top 10 Brands by Purchase Revenue\nCosmetics E-Commerce · October 2019",
        pad=16, color=PHILIPS_DEEP,
    )
    ax.set_xlabel("Total Revenue (USD)", labelpad=10, color=TEXT_PRIMARY)
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000:.0f}K")
    )
    ax.set_xlim(0, max_rev * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11, labelcolor=TEXT_PRIMARY)

    # Total revenue KPI badge (top-right)
    ax.text(
        0.98, 0.02,
        f"Top-10 Combined\n${plot_df['revenue'].sum():,.0f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10, fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=PHILIPS_DEEP,
            edgecolor=PHILIPS_ACCENT,
            linewidth=1.5,
        ),
    )

    # Footer
    fig.text(
        0.99, 0.005,
        "Source: Cosmetics E-Commerce Dataset | Philips BA Portfolio",
        ha="right", fontsize=8, color=TEXT_SECONDARY, style="italic",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved → {save_path}")


# =============================================================
# 7. ACTIONABLE BUSINESS INSIGHTS
# =============================================================
def print_insights(funnel: pd.DataFrame, brand_df: pd.DataFrame) -> None:
    """
    Derive and print 3 data-driven, executive-ready insights.

    Each insight follows the Situation → Data → Action framework
    commonly used in McKinsey-style business communications.
    """
    # --- Derived metrics ---
    view_sessions = funnel.loc[funnel["stage"] == "View",     "sessions"].values[0]
    cart_sessions = funnel.loc[funnel["stage"] == "Cart",     "sessions"].values[0]
    purch_sessions= funnel.loc[funnel["stage"] == "Purchase", "sessions"].values[0]

    view_to_cart  = cart_sessions  / view_sessions  * 100
    cart_to_purch = purch_sessions / cart_sessions  * 100
    overall_conv  = purch_sessions / view_sessions  * 100

    top_brand     = brand_df.iloc[0]["brand"]
    top_revenue   = brand_df.iloc[0]["revenue"]
    top_pct       = brand_df.iloc[0]["pct_of_total"]
    second_brand  = brand_df.iloc[1]["brand"]
    second_pct    = brand_df.iloc[1]["pct_of_total"]

    # Drop from view→cart vs cart→purchase — find the bigger leak
    bigger_drop_stage = "View → Cart" if view_to_cart < cart_to_purch else "Cart → Purchase"
    bigger_drop_rate  = min(view_to_cart, cart_to_purch)

    divider = "─" * 62

    print()
    print("=" * 62)
    print("  ACTIONABLE BUSINESS INSIGHTS  (Philips BA Portfolio)")
    print("=" * 62)

    print(f"\n  {divider}")
    print("  INSIGHT 1 — Critical Funnel Drop-Off: Prioritise Top-of-Funnel")
    print(f"  {divider}")
    print(
        f"  The steepest conversion loss occurs at the [{bigger_drop_stage}] stage,\n"
        f"  where only {bigger_drop_rate:.1f}% of sessions advance. The overall funnel\n"
        f"  conversion from Browse to Purchase stands at just {overall_conv:.2f}%.\n"
        f"  → Action: Deploy A/B-tested product-page optimisations (richer\n"
        f"    imagery, trust signals, social proof) to lift add-to-cart rate\n"
        f"    and recoup the highest-volume drop-off point first."
    )

    print(f"\n  {divider}")
    print("  INSIGHT 2 — Revenue Concentration Risk: Diversify Brand Mix")
    print(f"  {divider}")
    print(
        f"  '{top_brand.title()}' alone contributes {top_pct:.1f}% of top-10 revenue,\n"
        f"  while '{second_brand.title()}' adds {second_pct:.1f}%. Heavy concentration\n"
        f"  in 1–2 brands exposes the category to supply-chain & pricing risk.\n"
        f"  → Action: Negotiate co-marketing agreements with the #3–#5 brands\n"
        f"    to diversify revenue streams and reduce dependency on a single\n"
        f"    brand's promotional cycles."
    )

    print(f"\n  {divider}")
    print("  INSIGHT 3 — Cart Abandonment Recovery: High-ROI Quick Win")
    print(f"  {divider}")
    print(
        f"  {cart_sessions:,} sessions added items to cart but only\n"
        f"  {purch_sessions:,} completed a purchase — a {100 - cart_to_purch:.1f}% cart\n"
        f"  abandonment rate representing significant recoverable revenue.\n"
        f"  → Action: Implement a 3-step automated cart-recovery sequence\n"
        f"    (email at 1 h, push notification at 24 h, discount at 72 h).\n"
        f"    Industry benchmarks suggest 5–15% of abandoned carts can be\n"
        f"    recovered, translating to direct bottom-line impact."
    )

    print(f"\n  {'=' * 62}\n")


# =============================================================
# ENTRY POINT
# =============================================================
def main() -> None:
    """Orchestrate the full CRO analysis pipeline."""

    # -- 2. Load & clean --------------------------------------------------
    df = load_and_clean(
        filepath="data/2019-Oct.csv",
        nrows=1_000_000,
    )

    # -- 3. Funnel analysis -----------------------------------------------
    print("  Computing conversion funnel …")
    funnel = compute_funnel(df)
    print(funnel.to_string(index=False))
    print()

    # -- 4. Brand revenue -------------------------------------------------
    print("  Computing top-10 brand revenue …")
    brand_df, total_rev = compute_top_brands(df, top_n=10)
    print(brand_df[["brand", "revenue", "pct_of_total"]].to_string(index=False))
    print()

    # -- 5. Visualisation: funnel -----------------------------------------
    print("  Rendering charts …")
    plot_conversion_funnel(
        funnel,
        save_path=os.path.join(VISUALS_DIR, "conversion_funnel.png"),
    )

    # -- 6. Visualisation: brand revenue ----------------------------------
    plot_top_brands(
        brand_df,
        total_revenue=total_rev,
        save_path=os.path.join(VISUALS_DIR, "top_brands_revenue.png"),
    )

    # -- 7. Insights ------------------------------------------------------
    print_insights(funnel, brand_df)


if __name__ == "__main__":
    main()
