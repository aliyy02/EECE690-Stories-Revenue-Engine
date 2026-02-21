"""
Stories Coffee — Revenue Action Engine Dashboard

5-page Streamlit dashboard:
  0. Executive Overview (KPIs, top actions, action distribution)
  1. Revenue Levers (cross-sell, modifiers, size upsell, channels)
  2. Menu Engineering (product classification, kill list, bundles)
  3. Branch Playbooks (per-branch scorecards, branch grouping)
  4. Forecasting & Seasonal (sales forecast, seasonal patterns)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.revenue_analysis import (
    cross_sell_matrix,
    modifier_upsell,
    menu_bcg_matrix,
    bundle_builder,
    branch_playbooks,
    seasonal_strategy,
    channel_expansion,
    size_upsell,
)
from src.ml_models import sales_forecast, branch_clustering, menu_engineering
from src.action_engine import prioritize_actions, branch_action_cards
from src.utils import BRANCH_REGIONS

# ── Brand Colors ──────────────────────────────────────────────────────────────
C_GREEN = "#1B3A2D"
C_GOLD = "#C8A96E"
C_CREAM = "#F5F0EB"
C_RED = "#C0392B"
C_BLUE = "#2980B9"
C_GRAY = "#7F8C8D"
BCG_COLORS = {
    "Star": "#F1C40F",
    "Plow Horse": "#3498DB",
    "Puzzle": "#E67E22",
    "Dog": "#95A5A6",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=C_GREEN),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Data Loading (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    """Run all analyses and ML models. Cached for 1 hour."""
    from src.revenue_analysis import _load as load_data

    data = load_data()

    analyses = {
        "cross_sell": cross_sell_matrix(data),
        "modifier_upsell": modifier_upsell(data),
        "bcg": menu_bcg_matrix(data),
        "bundles": bundle_builder(data),
        "playbooks": branch_playbooks(data),
        "seasonal": seasonal_strategy(data),
        "channels": channel_expansion(data),
        "size_upsell": size_upsell(data),
    }

    ml = {
        "forecast": sales_forecast(data),
        "clusters": branch_clustering(data),
        "menu": menu_engineering(data),
    }

    actions_df = prioritize_actions(analyses)
    cards = branch_action_cards(analyses)

    return data, analyses, ml, actions_df, cards


# ── Helper Functions ─────────────────────────────────────────────────────────

def fmt(n, decimals=0):
    """Format large numbers with K/M suffixes."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:,.{decimals}f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:,.{decimals}f}K"
    return f"{n:,.{decimals}f}"


def info_box(title, text):
    """Render a styled explanation box for business context."""
    st.html(
        f"""<div style="background: white; padding: 14px 18px; margin: 10px 0 16px 0;
        border-radius: 8px; border: 1px solid {C_GOLD};">
        <span style="font-weight: 700; color: {C_GREEN}; font-size: 0.95rem;">
        {title}</span><br>
        <span style="font-size: 0.88rem; color: #444; line-height: 1.5;">
        {text}</span></div>"""
    )


def action_card(a, show_branch=True):
    """Render a single action as a styled card."""
    color = C_RED if a.get("priority") == "HIGH" else C_GOLD
    branch = a.get("branch", a.get("product", a.get("bundle", a.get("season", ""))))
    header = f"<b>{branch}</b>" if show_branch and branch else ""
    priority_label = "HIGH — Act this week" if a.get("priority") == "HIGH" else "MEDIUM — Plan this month"
    st.html(
        f"""<div style="border-left: 4px solid {color}; padding: 10px 14px; margin: 6px 0;
        background: white; border-radius: 0 6px 6px 0;">
        <span style="color: {color}; font-weight: 700; font-size: 0.75rem;">
        {priority_label}</span>
        {f' &mdash; {header}' if header else ''}
        <br><span style="font-size: 0.88rem; color: #333;">{a.get('insight', '')}</span>
        <br><span style="font-size: 0.88rem; color: {C_GREEN}; font-weight: 600;">
        Action: {a.get('action', '')}</span></div>"""
    )


def priority_legend():
    """Show the priority color legend."""
    st.html(
        f"""<div style="display: flex; gap: 24px; margin: 8px 0 16px 0; font-size: 0.85rem;">
        <span><span style="color: {C_RED}; font-weight: 700;">HIGH</span> = Act this week (biggest revenue impact)</span>
        <span><span style="color: {C_GOLD}; font-weight: 700;">MEDIUM</span> = Plan this month (solid opportunity)</span>
        </div>"""
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═════════════════════════════════════════════════════════════════════════════


def page_overview(analyses, ml, actions_df, cards):
    """Executive overview with KPIs and top actions."""
    st.header("Executive Overview")

    st.markdown(
        "This dashboard analyzes **full-year 2025 sales data** across all 25 Stories Coffee branches "
        "to surface **specific, actionable recommendations** that can drive revenue growth. "
        "Every insight below is paired with a concrete action your team can implement."
    )

    total_actions = len(actions_df)
    high_actions = len(actions_df[actions_df["priority"] == "HIGH"])
    cs = analyses["cross_sell"]
    mu = analyses["modifier_upsell"]
    su = analyses["size_upsell"]

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Recommendations", total_actions)
    k2.metric("Urgent Actions", high_actions)
    k3.metric("Food Attach Rate", f"{cs['avg_attach_rate']:.0f}%")
    k4.metric("Modifiers per 100 Drinks", f"{mu['avg_attach_rate']:.0f}")
    k5.metric("Size Upsell Potential", fmt(su["upsell_10pct_revenue"]))

    with st.expander("What do these numbers mean?"):
        st.markdown(f"""
| Metric | Meaning |
|--------|---------|
| **Recommendations** | Total number of specific actions generated by our analysis |
| **Urgent Actions** | HIGH priority actions with the biggest potential revenue impact — address these first |
| **Food Attach Rate** | For every 100 drink orders, how many also include a food item (chain average: {cs['avg_attach_rate']:.0f}%). Higher = more cross-selling |
| **Modifiers per 100 Drinks** | How many paid add-ons (extra shot, oat milk, etc.) are sold per 100 drink orders. Values above 100 mean customers often add 2+ extras per drink (chain avg: {mu['avg_attach_rate']:.0f}) |
| **Size Upsell Potential** | Estimated additional revenue if just 10% of small/medium drink customers upgraded to the next size |
        """)

    st.divider()

    # Priority legend + HIGH actions by source
    st.subheader("Top Urgent Actions")
    priority_legend()

    high_df = actions_df[actions_df["priority"] == "HIGH"]
    for source in high_df["source"].unique():
        src_df = high_df[high_df["source"] == source]
        with st.expander(f"**{source}** ({len(src_df)} urgent actions)", expanded=False):
            for _, row in src_df.head(5).iterrows():
                action_card(row.to_dict())

    # Action distribution chart
    st.divider()
    st.subheader("All Recommendations by Category")
    st.caption("How many recommendations were generated from each type of analysis, colored by urgency.")
    source_counts = actions_df.groupby(["source", "priority"]).size().reset_index(name="count")
    fig = px.bar(
        source_counts,
        x="source",
        y="count",
        color="priority",
        color_discrete_map={"HIGH": C_RED, "MEDIUM": C_GOLD, "LOW": C_GRAY},
        barmode="stack",
        labels={"source": "Analysis Type", "count": "Number of Recommendations", "priority": "Priority"},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def page_revenue_levers(analyses):
    """Revenue Levers: cross-sell, modifiers, size upsell, channels."""
    st.header("Revenue Levers")

    st.markdown(
        "These four strategies represent the **quickest wins** for Stories Coffee. "
        "Each tab identifies branches where a specific behavior change — staff training, "
        "menu redesign, or delivery onboarding — can increase revenue **without adding new customers**."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Food Cross-Sell", "Drink Add-Ons", "Size Upgrades", "Delivery Channels"]
    )

    # ── Tab 1: Cross-Sell ─────────────────────────────────────────────────
    with tab1:
        cs = analyses["cross_sell"]

        info_box(
            "What is Food Attachment Rate?",
            "This measures how often customers add a food item to their drink order. "
            f"For example, if a branch sells 1,000 drinks and 500 food items, its food attach rate is 50%. "
            f"The chain average is <b>{cs['avg_attach_rate']:.0f}%</b>. "
            "Branches below average have untapped cross-sell potential — staff can be trained to suggest "
            "pastries in the morning and cookies/donuts in the afternoon."
        )

        st.subheader("Food Attachment Rate by Branch")

        df = cs["data"].copy()
        fig = px.bar(
            df.sort_values("food_attach_rate"),
            x="food_attach_rate",
            y="branch",
            orientation="h",
            color="region",
            labels={"food_attach_rate": "Food Attach Rate (%)", "branch": "", "region": "Region"},
        )
        fig.add_vline(
            x=cs["avg_attach_rate"], line_dash="dash", line_color=C_RED,
            annotation_text=f"Chain Avg: {cs['avg_attach_rate']:.0f}%",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=600,
                          legend=dict(title="Region", orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The red dashed line marks the chain average. Branches to its left are cross-sell opportunities.")

        # Revenue vs attach scatter
        st.subheader("Revenue vs Food Attachment")
        st.caption("Each bubble is a branch. Bigger bubble = more food revenue. Branches in the bottom-right are high-revenue but low food-attach — biggest opportunities.")
        fig2 = px.scatter(
            df,
            x="bev_revenue",
            y="food_attach_rate",
            size="food_revenue",
            color="region",
            hover_name="branch",
            labels={"bev_revenue": "Beverage Revenue", "food_attach_rate": "Food Attach Rate (%)", "region": "Region"},
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recommended Actions")
        priority_legend()
        for a in cs["actions"][:8]:
            action_card(a)

    # ── Tab 2: Modifier Upsell ────────────────────────────────────────────
    with tab2:
        mu = analyses["modifier_upsell"]

        info_box(
            "What are Modifiers (Add-Ons)?",
            "Modifiers are paid extras customers add to their drink — extra espresso shot, oat milk, "
            "flavored syrup, etc. Each modifier adds revenue at <b>near-100% margin</b> since the cost "
            f"is minimal. The chain averages <b>{mu['avg_attach_rate']:.0f} modifiers per 100 drinks</b>. "
            "Rates above 100% mean popular drinks often get 2+ add-ons — this is healthy and shows "
            "strong upselling culture. "
            "Branches with low modifier rates need better staff prompting: <i>'Would you like an extra shot today?'</i>"
        )

        st.subheader("Modifiers per 100 Drinks by Branch")

        bs = mu["branch_stats"].copy()
        fig = px.bar(
            bs.sort_values("mod_attach_rate"),
            x="mod_attach_rate",
            y="branch",
            orientation="h",
            labels={"mod_attach_rate": "Modifiers per 100 Drinks", "branch": ""},
            color_discrete_sequence=[C_GOLD],
        )
        fig.add_vline(
            x=mu["avg_attach_rate"], line_dash="dash", line_color=C_RED,
            annotation_text=f"Chain Avg: {mu['avg_attach_rate']:.0f}",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Branches to the left of the red line are under-selling add-ons relative to the chain.")

        # Top paid modifiers
        st.subheader("Most Popular Paid Add-Ons (Chain-wide)")
        st.caption("These are the add-ons customers order most often. Use this to guide staff recommendations.")
        paid = mu["top_paid_modifiers"].reset_index()
        if not paid.empty:
            fig3 = px.bar(
                paid.head(10),
                x="qty",
                y="product",
                orientation="h",
                color_discrete_sequence=[C_GREEN],
                labels={"qty": "Units Sold (2025)", "product": ""},
            )
            fig3.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Recommended Actions")
        priority_legend()
        for a in mu["actions"][:8]:
            action_card(a)

    # ── Tab 3: Size Upsell ────────────────────────────────────────────────
    with tab3:
        su = analyses["size_upsell"]

        info_box(
            "What is Size Upsell?",
            "Larger drink sizes have higher profit margins — a Large costs only slightly more to make than a Small, "
            "but the customer pays significantly more. This analysis shows the chain-wide size distribution and "
            "estimates how much revenue would grow if a small percentage of customers chose one size up. "
            "Staff can be trained to suggest: <i>'Would you like a Medium? It's only [X] more.'</i>"
        )

        st.subheader("Chain-wide Size Distribution")

        sd = su["size_dist"].reset_index()
        col1, col2 = st.columns([1, 1])

        with col1:
            fig = px.pie(
                sd,
                values="qty",
                names="size",
                color="size",
                color_discrete_map={"SMALL": C_GOLD, "MEDIUM": C_BLUE, "LARGE": C_GREEN},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Price Ladder by Size")
            display_sd = sd[["size", "qty", "avg_price", "avg_profit", "pct_of_total"]].copy()
            display_sd["qty"] = display_sd["qty"].apply(lambda x: f"{x:,.0f}")
            display_sd["avg_price"] = display_sd["avg_price"].apply(lambda x: f"{x:,.0f}")
            display_sd["avg_profit"] = display_sd["avg_profit"].apply(lambda x: f"{x:,.0f}")
            display_sd["pct_of_total"] = display_sd["pct_of_total"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(
                display_sd.rename(columns={
                    "size": "Size", "qty": "Units Sold", "avg_price": "Avg Price",
                    "avg_profit": "Profit per Unit", "pct_of_total": "Share of Orders",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.metric(
            "Estimated Revenue Gain from 10% Size Upgrades",
            fmt(su["upsell_10pct_revenue"]),
            help="If 10% of Small customers chose Medium, and 10% of Medium chose Large, this is the additional revenue.",
        )

        # Branch large% chart
        st.subheader("Large Size Adoption by Branch")
        st.caption("Higher % means the branch is already upselling well. Lower % = more room for improvement.")
        bsz = su["branch_size"]
        fig2 = px.bar(
            bsz.sort_values("large_pct"),
            x="large_pct",
            y="branch",
            orientation="h",
            labels={"large_pct": "% of Orders That Are Large", "branch": ""},
            color_discrete_sequence=[C_GREEN],
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=600)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recommended Actions")
        priority_legend()
        for a in su["actions"]:
            action_card(a)

    # ── Tab 4: Channel Mix ────────────────────────────────────────────────
    with tab4:
        ch = analyses["channels"]

        info_box(
            "What are Service Channels?",
            "Revenue comes from three channels: <b>Dine-In</b> (customers eating at the table), "
            "<b>Takeaway</b> (counter pickup), and <b>Delivery</b> (via Toters app). "
            f"Currently, <b>{len(ch['no_toters'])} of 25 branches</b> have little or no delivery presence. "
            f"Where Toters is active, it averages <b>{ch['avg_toters_pct']:.1f}%</b> of branch revenue — "
            "a massive untapped channel for most locations."
        )

        st.subheader("Revenue by Service Channel")
        st.caption("Each bar is a branch, split by how revenue comes in. Branches with no blue (Toters) are missing delivery revenue.")

        mix = ch["channel_mix"].copy()
        svc_cols = {}
        for c in mix.columns:
            if c.endswith("_revenue") and c not in ("total_revenue", "toters_revenue"):
                svc_cols[c] = c.replace("_revenue", "").replace("_", " ").title()
        if "toters_revenue" in mix.columns:
            svc_cols["toters_revenue"] = "Toters (Delivery)"

        fig = go.Figure()
        colors = [C_GREEN, C_GOLD, C_BLUE, C_RED]
        for i, (col, label) in enumerate(svc_cols.items()):
            fig.add_trace(go.Bar(
                name=label, x=mix["branch"], y=mix[col],
                marker_color=colors[i % len(colors)],
            ))
        fig.update_layout(
            barmode="stack",
            xaxis_title="", yaxis_title="Revenue",
            legend=dict(title="Channel", orientation="h", y=-0.2),
            **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Branches with No Delivery", len(ch["no_toters"]),
                       help="Number of branches with less than 1% of revenue from Toters delivery")
        with col2:
            st.metric("Avg Delivery Share (where active)", f"{ch['avg_toters_pct']:.1f}%",
                       help="Among branches that use Toters, this is the average share of total revenue from delivery")

        st.subheader("Recommended Actions — Delivery Expansion")
        priority_legend()
        for a in ch["actions"][:8]:
            action_card(a)


def page_menu_engineering(analyses, ml):
    """Menu Engineering: product classification, kill list, bundles."""
    st.header("Menu Engineering")

    st.markdown(
        "We analyzed **every product** on the Stories Coffee menu to determine which items are earning "
        "their spot and which are dragging down performance. Products are classified into four groups "
        "based on how many units they sell (volume) and how much profit each sale generates (margin)."
    )

    tab1, tab2, tab3 = st.tabs(["Product Classification", "Stars & Kill List", "Combo Builder"])

    # ── Tab 1: BCG Scatter ────────────────────────────────────────────────
    with tab1:
        me = ml["menu"]
        menu_df = me["menu"].copy()

        info_box(
            "How to Read This Chart",
            "Every dot represents a menu item. The chart divides products into <b>four groups</b>:<br><br>"
            "<span style='color:#F1C40F;font-weight:700;'>&#9679; Stars</span> (top-right): "
            "High sales + high profit margin. <b>Protect these</b> — never discount them.<br>"
            "<span style='color:#3498DB;font-weight:700;'>&#9679; Plow Horses</span> (bottom-right): "
            "High sales but low margin. Consider a <b>small price increase</b> — demand is strong.<br>"
            "<span style='color:#E67E22;font-weight:700;'>&#9679; Puzzles</span> (top-left): "
            "High margin but low sales. <b>Hidden gems</b> — promote these with better menu placement.<br>"
            "<span style='color:#95A5A6;font-weight:700;'>&#9679; Dogs</span> (bottom-left): "
            "Low sales + low margin. Candidates for <b>removal</b> from the menu.<br><br>"
            "Bubble size shows total profit. Dashed lines mark the midpoint for each axis."
        )

        st.subheader("Product Classification Map")

        # Category filter
        cat_filter = st.selectbox("Filter by Category", ["All Products", "Beverages Only", "Food Only"], key="bcg_cat")
        if cat_filter == "Beverages Only":
            menu_df = menu_df[menu_df["category"] == "BEVERAGES"]
        elif cat_filter == "Food Only":
            menu_df = menu_df[menu_df["category"] == "FOOD"]

        # Plotly size must be >= 0; use absolute profit so loss-makers still show
        menu_df["bubble_size"] = menu_df["profit"].abs().clip(lower=1)

        fig = px.scatter(
            menu_df,
            x="qty",
            y="profit_margin",
            size="bubble_size",
            color="bcg_class",
            hover_name="product",
            hover_data={
                "division": True,
                "menu_power": ":.0f",
                "n_branches": True,
                "qty": ":,.0f",
                "profit_margin": ":.1f",
                "profit": ":,.0f",
                "bubble_size": False,
            },
            color_discrete_map=BCG_COLORS,
            labels={
                "qty": "Units Sold (2025)",
                "profit_margin": "Profit Margin (%)",
                "bcg_class": "Classification",
                "division": "Division",
                "menu_power": "Menu Power Score",
                "n_branches": "Sold at # Branches",
                "profit": "Total Profit",
                "bubble_size": "Bubble Size",
            },
        )
        # Median lines
        qty_med = menu_df["qty"].median()
        margin_med = menu_df["profit_margin"].median()
        fig.add_vline(x=qty_med, line_dash="dash", line_color=C_GRAY, opacity=0.4)
        fig.add_hline(y=margin_med, line_dash="dash", line_color=C_GRAY, opacity=0.4)

        # Quadrant labels
        fig.add_annotation(x=menu_df["qty"].max() * 0.8, y=95, text="STARS",
                           showarrow=False, font=dict(size=16, color=BCG_COLORS["Star"], family="Inter"))
        fig.add_annotation(x=menu_df["qty"].max() * 0.8, y=15, text="PLOW HORSES",
                           showarrow=False, font=dict(size=16, color=BCG_COLORS["Plow Horse"], family="Inter"))
        fig.add_annotation(x=menu_df["qty"].min() + 100, y=95, text="PUZZLES",
                           showarrow=False, font=dict(size=16, color=BCG_COLORS["Puzzle"], family="Inter"))
        fig.add_annotation(x=menu_df["qty"].min() + 100, y=15, text="DOGS",
                           showarrow=False, font=dict(size=16, color=BCG_COLORS["Dog"], family="Inter"))

        fig.update_layout(**PLOTLY_LAYOUT, height=600,
                          legend=dict(title="Classification"))
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Classification Summary")
        st.caption("How many products fall into each group, and their combined performance.")
        summary = me["summary"].reset_index()
        summary_display = summary.copy()
        summary_display = summary_display.rename(columns={
            "bcg_class": "Group", "count": "Products", "total_qty": "Total Units",
            "total_revenue": "Total Revenue", "total_profit": "Total Profit",
            "avg_margin": "Avg Margin %", "avg_menu_power": "Avg Menu Power",
        })
        for col in ["Total Units", "Total Revenue", "Total Profit"]:
            if col in summary_display.columns:
                summary_display[col] = summary_display[col].apply(lambda x: fmt(x))
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        # Menu Power explanation + histogram
        with st.expander("What is Menu Power Score?"):
            st.markdown("""
**Menu Power** is a composite score from 0 to 100 that ranks each product on four dimensions:

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| Volume | 25% | Units sold compared to other products in the same category |
| Margin | 25% | Profit margin compared to peers |
| Profit Contribution | 30% | Total profit generated (bigger impact = higher score) |
| Branch Coverage | 20% | How many branches carry this product (wider = more important) |

**Interpretation**: Products scoring 70+ are strong performers. Below 30 should be reviewed for removal.
            """)

        fig2 = px.histogram(
            me["menu"],
            x="menu_power",
            color="bcg_class",
            color_discrete_map=BCG_COLORS,
            nbins=30,
            labels={"menu_power": "Menu Power Score (0-100)", "bcg_class": "Classification"},
        )
        fig2.update_layout(**PLOTLY_LAYOUT, title="Distribution of Menu Power Scores")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Technical Details: How Product Classification Works"):
            st.markdown("""
This analysis uses the **BCG (Boston Consulting Group) Matrix**, a widely-used business framework
adapted here for menu items.

**How it works:**
1. For each product, we calculate total quantity sold and profit margin (%)
2. We find the median (middle value) for both quantity and margin within each category (Beverages and Food separately)
3. Products are placed into quadrants based on whether they are above or below the median on each dimension

**Menu Power Score** is a weighted composite score that combines volume, margin, total profit
contribution, and how many branches carry the product. This gives a single number (0-100)
that summarizes overall product health.
            """)

    # ── Tab 2: Kill List & Stars ──────────────────────────────────────────
    with tab2:
        bcg = analyses["bcg"]

        st.markdown(
            "**Stars** are your best-performing products — high sales and high margins. Never discount them. "
            "**Kill List** items waste menu space, slow down kitchen operations, and confuse customers. "
            "Removing them simplifies training and reduces ingredient waste. "
            "**Puzzles** are hidden gems with great margins but low awareness — promote them."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stars — Protect These")
            st.caption("High volume + high margin. Your best products.")
            stars = bcg["stars"].head(15)[["product", "category", "qty", "profit", "profit_margin"]].copy()
            stars["qty"] = stars["qty"].apply(lambda x: f"{x:,.0f}")
            stars["profit"] = stars["profit"].apply(lambda x: fmt(x))
            stars["profit_margin"] = stars["profit_margin"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(stars.rename(columns={
                "product": "Product", "category": "Category", "qty": "Units Sold",
                "profit": "Total Profit", "profit_margin": "Margin",
            }), use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Kill List — Consider Removing")
            st.caption("Low volume + low margin. These drag down performance.")
            dogs = bcg["kill_list"].head(15)[["product", "category", "qty", "profit", "profit_margin"]].copy()
            dogs["qty"] = dogs["qty"].apply(lambda x: f"{x:,.0f}")
            dogs["profit"] = dogs["profit"].apply(lambda x: fmt(x))
            dogs["profit_margin"] = dogs["profit_margin"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(dogs.rename(columns={
                "product": "Product", "category": "Category", "qty": "Units Sold",
                "profit": "Total Profit", "profit_margin": "Margin",
            }), use_container_width=True, hide_index=True)

        st.subheader("Puzzles — Hidden Gems to Promote")
        st.caption("High margin but low sales. Better menu placement, staff training, and social media features can unlock their potential.")
        puzzles = bcg["puzzles"].head(10)[["product", "category", "qty", "profit", "profit_margin"]].copy()
        puzzles["qty"] = puzzles["qty"].apply(lambda x: f"{x:,.0f}")
        puzzles["profit"] = puzzles["profit"].apply(lambda x: fmt(x))
        puzzles["profit_margin"] = puzzles["profit_margin"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(puzzles.rename(columns={
            "product": "Product", "category": "Category", "qty": "Units Sold",
            "profit": "Total Profit", "profit_margin": "Margin",
        }), use_container_width=True, hide_index=True)

        st.subheader("Recommended Actions")
        priority_legend()
        for a in bcg["actions"][:10]:
            action_card(a, show_branch=False)

    # ── Tab 3: Bundle Builder ─────────────────────────────────────────────
    with tab3:
        bu = analyses["bundles"]

        info_box(
            "How Combo Suggestions Work",
            "We paired the <b>highest-selling beverages</b> with the <b>highest-margin food items</b> "
            "and applied an 8% discount. Even after the discount, these combos maintain strong margins "
            "while encouraging customers to add food to their order (cross-sell). "
            "Feature these on menu boards and counter displays."
        )

        st.subheader("Top Suggested Combos")

        bundles = bu["bundles"].head(15).copy()
        bundles["individual_price"] = bundles["individual_price"].apply(lambda x: fmt(x))
        bundles["bundle_price"] = bundles["bundle_price"].apply(lambda x: fmt(x))
        bundles["bundle_margin"] = bundles["bundle_margin"].apply(lambda x: f"{x:.1f}%")
        bundles["bev_qty"] = bundles["bev_qty"].apply(lambda x: f"{x:,.0f}")

        st.dataframe(
            bundles[["beverage", "food", "individual_price", "bundle_price", "bundle_margin", "bev_qty"]].rename(
                columns={
                    "beverage": "Drink", "food": "Food Item",
                    "individual_price": "Bought Separately", "bundle_price": "Combo Price (8% off)",
                    "bundle_margin": "Combo Margin", "bev_qty": "Drink Sales Volume",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Combo Margin = profit percentage after applying the 8% bundle discount. Higher = more profitable combo.")

        st.subheader("Recommended Actions")
        priority_legend()
        for a in bu["actions"]:
            action_card(a, show_branch=False)


def page_branch_playbooks(analyses, ml, cards):
    """Branch Playbooks: per-branch scorecards, branch grouping."""
    st.header("Branch Playbooks")

    st.markdown(
        "Each branch gets a **performance scorecard** comparing it to the chain average. "
        "Select a branch to see its strengths, weaknesses, and specific improvement actions. "
        "The **Branch Groups** tab shows branches clustered by similar performance characteristics."
    )

    tab1, tab2 = st.tabs(["Branch Scorecards", "Branch Groups"])

    # ── Tab 1: Branch Cards ───────────────────────────────────────────────
    with tab1:
        pb = analyses["playbooks"]
        summary = pb["summary"].copy()

        branch = st.selectbox("Select a Branch", sorted(summary["branch"].tolist()), key="pb_branch")
        card = cards.get(branch, {})

        if card:
            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Revenue (2025)", fmt(card["revenue"]))
            c2.metric("Profit Margin", f"{card['profit_margin']:.1f}%",
                       help="Percentage of revenue kept as profit after costs")
            c3.metric("Food Attach", f"{card['food_attach_rate']:.1f}%",
                       help="Food items sold per 100 beverage orders")
            c4.metric("Add-Ons / 100 Drinks", f"{card['mod_attach_rate']:.1f}",
                       help="Paid modifiers per 100 drink orders. Above 100 = multiple add-ons per drink on average.")
            c5.metric("Delivery %", f"{card['toters_pct']:.1f}%",
                       help="Share of revenue from Toters delivery")

            st.caption(
                f"**Region**: {card['region']} | **Best Month**: {card['peak_month']} | "
                f"**Recommendations**: {card['total_actions']} ({card['high_priority_actions']} urgent)"
            )

            with st.expander("What do these metrics mean?"):
                st.markdown("""
| Metric | What It Measures | What "Good" Looks Like |
|--------|-----------------|----------------------|
| **Revenue** | Total sales for all of 2025 | Higher = busier branch |
| **Profit Margin** | % of revenue kept as profit | 70%+ is strong for coffee |
| **Food Attach** | Food items per 100 drinks | 50%+ is good cross-selling |
| **Add-Ons / 100 Drinks** | Paid extras per 100 drinks | 100+ means strong upselling (2+ add-ons per drink) |
| **Delivery %** | Revenue from Toters app | Depends on location; 0% = untapped |
                """)

            # Comparison chart
            avgs = pb["chain_averages"]
            comparison = pd.DataFrame({
                "Metric": ["Food Attach Rate", "Add-Ons / 100 Drinks", "Delivery %", "Profit Margin"],
                "This Branch": [card["food_attach_rate"], card["mod_attach_rate"], card["toters_pct"], card["profit_margin"]],
                "Chain Average": [avgs["food_attach_rate"], avgs["mod_attach_rate"], avgs["toters_pct"], avgs["profit_margin"]],
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=f"{branch}", x=comparison["Metric"], y=comparison["This Branch"],
                marker_color=C_GREEN,
            ))
            fig.add_trace(go.Bar(
                name="Chain Average", x=comparison["Metric"], y=comparison["Chain Average"],
                marker_color=C_GOLD,
            ))
            fig.update_layout(
                barmode="group",
                title=f"{branch} vs Chain Average",
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", y=-0.15),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green bars above gold = outperforming. Green below gold = improvement opportunity.")

            # Branch actions
            if card["actions"]:
                st.subheader("Recommended Actions for This Branch")
                priority_legend()
                for a in card["actions"]:
                    action_card(a)
        else:
            st.info("No scorecard data available for this branch.")

        # All branches table
        st.divider()
        st.subheader("All Branches — Comparison Table")
        st.caption("Compare key metrics across all branches. Use this to identify which branches need the most attention.")
        display_cols = ["branch", "region", "total_revenue", "profit_margin",
                        "food_attach_rate", "mod_attach_rate", "toters_pct", "peak_month"]
        display_df = summary[display_cols].copy()
        display_df["total_revenue"] = display_df["total_revenue"].apply(lambda x: fmt(x))
        display_df["profit_margin"] = display_df["profit_margin"].apply(lambda x: f"{x:.1f}%")
        display_df["food_attach_rate"] = display_df["food_attach_rate"].apply(lambda x: f"{x:.1f}%")
        display_df["mod_attach_rate"] = display_df["mod_attach_rate"].apply(lambda x: f"{x:.1f}")
        display_df["toters_pct"] = display_df["toters_pct"].apply(lambda x: f"{x:.1f}%")
        display_df = display_df.rename(columns={
            "branch": "Branch", "region": "Region", "total_revenue": "Revenue",
            "profit_margin": "Margin", "food_attach_rate": "Food Attach",
            "mod_attach_rate": "Add-Ons/100 Drinks", "toters_pct": "Delivery %",
            "peak_month": "Best Month",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Clusters ───────────────────────────────────────────────────
    with tab2:
        cl = ml["clusters"]
        features = cl["features"].copy()

        info_box(
            "What are Branch Groups?",
            f"We analyzed all branches across 8 performance dimensions (revenue, margins, food attachment, "
            f"delivery usage, etc.) and identified <b>{len(cl['cluster_labels'])} natural groups</b> of "
            "branches that behave similarly. Branches in the same group should receive similar strategies."
        )

        st.subheader("Branch Groups — Performance Map")
        st.caption("Each dot is a branch. Dot size = average transaction value. Color = assigned group. "
                    "Hover over any dot to see detailed metrics.")

        fig = px.scatter(
            features,
            x="total_revenue",
            y="food_attach_rate",
            size="avg_ticket",
            color="cluster_label",
            hover_name="branch",
            hover_data={
                "region": True,
                "bev_profit_pct": ":.1f",
                "toters_pct": ":.1f",
                "table_pct": ":.1f",
            },
            labels={
                "total_revenue": "Total Revenue (2025)",
                "food_attach_rate": "Food Attach Rate (%)",
                "cluster_label": "Branch Group",
                "region": "Region",
                "bev_profit_pct": "Beverage Margin %",
                "toters_pct": "Delivery %",
                "table_pct": "Dine-In %",
            },
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=500,
                          legend=dict(title="Branch Group"))
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles
        st.subheader("Group Profiles — Average Metrics")
        st.caption("Average performance metrics for each group. Use this to design group-level strategies.")
        profiles = cl["cluster_profiles"].copy()
        profiles.index = profiles.index.map(cl["cluster_labels"])
        profiles_display = profiles.round(1).copy()
        profiles_display = profiles_display.rename(columns={
            "total_revenue": "Avg Revenue", "food_share": "Food Share %",
            "bev_profit_pct": "Bev Margin %", "food_profit_pct": "Food Margin %",
            "food_attach_rate": "Food Attach %", "avg_ticket": "Avg Ticket",
            "toters_pct": "Delivery %", "table_pct": "Dine-In %",
        })
        st.dataframe(profiles_display, use_container_width=True)

        # Cluster actions
        st.subheader("Recommended Group-Level Strategies")
        for a in cl["actions"]:
            branches = ", ".join(a.get("branches", [])[:6])
            remaining = len(a.get("branches", [])) - 6
            branch_text = branches + (f", +{remaining} more" if remaining > 0 else "")
            st.html(
                f"""<div style="border-left: 4px solid {C_GOLD}; padding: 12px 14px; margin: 8px 0;
                background: white; border-radius: 0 6px 6px 0;">
                <span style="font-weight: 700; color: {C_GREEN}; font-size: 0.95rem;">{a['cluster']}</span><br>
                <span style="font-size: 0.88rem; color: #333;">{a['insight']}</span><br>
                <span style="font-size: 0.88rem; color: {C_GREEN}; font-weight: 600;">
                Action: {a['action']}</span><br>
                <span style="font-size: 0.82rem; color: {C_GRAY};">Branches: {branch_text}</span>
                </div>"""
            )

        with st.expander("Technical Details: How Branch Grouping Works"):
            st.markdown("""
**Method**: K-Means Clustering — a statistical technique that groups data points (branches)
into segments where members are more similar to each other than to other groups.

**What we measured for each branch:**
- Total revenue, food revenue share, beverage margin %, food margin %
- Food attachment rate, average transaction value
- Delivery (Toters) share, dine-in (table) share

**Process:**
1. All metrics are normalized so no single measure dominates the grouping
2. The algorithm tests different numbers of groups (2 through 7)
3. We select the number that provides meaningful distinctions without over-splitting
4. Each branch is assigned to its closest group based on overall similarity

The chart below shows how model quality changes with different group counts.
The "elbow" (where the curve bends) suggests the optimal number of groups.
            """)

            # Elbow plot inside the technical expander
            inertias = cl["inertias"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(inertias.keys()),
                y=list(inertias.values()),
                mode="lines+markers",
                line=dict(color=C_GREEN, width=2),
                marker=dict(size=8),
            ))
            fig2.update_layout(
                title="Model Quality by Number of Groups",
                xaxis_title="Number of Groups",
                yaxis_title="Within-Group Variation (lower = tighter groups)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig2, use_container_width=True)


def page_forecasting(analyses, ml):
    """Forecasting & Seasonal: sales forecast, seasonal patterns."""
    st.header("Forecasting & Seasonal Strategy")

    st.markdown(
        "This page uses **2025 sales patterns** to predict upcoming months and highlights "
        "seasonal trends your team should plan around for staffing, inventory, and promotions."
    )

    tab1, tab2 = st.tabs(["Sales Forecast", "Seasonal Patterns"])

    # ── Tab 1: Forecast ───────────────────────────────────────────────────
    with tab1:
        fc = ml["forecast"]

        st.warning(
            "**Forecast Disclaimer**: This model was trained on only 1 year of data (2025) "
            "with in-sample evaluation. The R\u00b2 score reflects how well the model fits "
            "training data, not true out-of-sample accuracy. Use these forecasts for "
            "**directional planning** (staffing, inventory), not financial commitments."
        )

        info_box(
            "How the Forecast Works",
            "We built a statistical model that learned from 2025 monthly sales patterns — "
            "which months are strong, which are weak, how long each branch has been open, etc. "
            "It then predicts what Jan\u2013Jun 2026 sales will look like. "
            "Use these predictions to <b>plan staffing and inventory</b> ahead of time, "
            "but treat them as directional estimates, not guarantees."
        )

        st.subheader("Monthly Sales Forecast")

        accuracy_pct = fc["train_r2"] * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Accuracy", f"{accuracy_pct:.0f}%",
                   help="How much of sales variation the model explains. 100% = perfect prediction. Above 60% is useful.")
        c2.metric("Avg Prediction Error", f"\u00B1{fmt(fc['train_mae'])}",
                   help="On average, predictions are off by this amount in either direction")
        c3.metric("Branches Forecasted", len(fc["eligible_branches"]),
                   help="Only branches with 6+ months of data are forecasted")

        st.caption(
            f"The model explains **{accuracy_pct:.0f}%** of monthly sales variation. "
            f"Predictions are typically within **\u00B1{fmt(fc['train_mae'])}** of actual values."
        )

        # Actuals + forecast chart
        from src.revenue_analysis import _load as load_data
        data_raw = load_data()
        monthly = data_raw["monthly"]
        m25 = monthly[monthly["year"] == 2025].copy()

        branch_filter = st.selectbox(
            "View Branch",
            ["All Branches (combined)"] + sorted(fc["eligible_branches"]),
            key="fc_branch",
        )

        if branch_filter == "All Branches (combined)":
            actuals = m25.groupby("month_num")["sales"].sum().reset_index()
            forecast = fc["forecast"].groupby("month_num")["predicted_sales"].sum().reset_index()
        else:
            actuals = m25[m25["branch"] == branch_filter][["month_num", "sales"]]
            forecast = fc["forecast"][fc["forecast"]["branch"] == branch_filter][
                ["month_num", "predicted_sales"]
            ]

        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[month_names[m] for m in actuals["month_num"]],
            y=actuals["sales"],
            mode="lines+markers",
            name="2025 Actual Sales",
            line=dict(color=C_GREEN, width=3),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=[month_names[m] for m in forecast["month_num"]],
            y=forecast["predicted_sales"],
            mode="lines+markers",
            name="2026 Forecast",
            line=dict(color=C_GOLD, width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
        ))

        # Jan 2026 actual
        jan26 = fc["actuals_jan26"]
        if not jan26.empty:
            if branch_filter == "All Branches (combined)":
                jan_val = jan26["sales"].sum()
            else:
                jan_b = jan26[jan26["branch"] == branch_filter]
                jan_val = jan_b["sales"].sum() if not jan_b.empty else None
            if jan_val:
                fig.add_trace(go.Scatter(
                    x=["Jan"], y=[jan_val],
                    mode="markers",
                    name="Jan 2026 Actual (validation)",
                    marker=dict(size=14, color=C_RED, symbol="star"),
                ))

        fig.update_layout(
            title=f"Sales Trend & Forecast — {branch_filter}",
            xaxis_title="Month", yaxis_title="Sales Revenue",
            legend=dict(orientation="h", y=-0.15),
            **PLOTLY_LAYOUT, height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Solid green line = actual 2025 data. Dashed gold line = 2026 predictions. Red star = Jan 2026 actual for comparison.")

        # Feature importance
        with st.expander("What drives the forecast? (Factor Importance)"):
            st.markdown(
                "The chart below shows which factors have the strongest influence on sales predictions. "
                "Positive values push sales **up**, negative values push sales **down**."
            )
            imp = fc["feature_importance"].reset_index()
            imp.columns = ["Factor", "Influence"]
            # Rename technical features to plain English
            rename_map = {
                "is_seasonal_low": "Seasonal Low (May-Jun)",
                "is_summer": "Summer (Jul-Aug)",
                "is_q4": "Q4 Holidays (Oct-Dec)",
                "month_sin": "Month Cycle (sin)",
                "month_cos": "Month Cycle (cos)",
                "months_active": "Branch Maturity",
            }
            imp["Factor"] = imp["Factor"].map(lambda x: rename_map.get(x, x))
            fig2 = px.bar(
                imp, x="Influence", y="Factor", orientation="h",
                color_discrete_sequence=[C_GREEN],
                labels={"Influence": "Impact on Sales (positive = increases sales)"},
            )
            fig2.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Technical Details: Forecast Model"):
            st.markdown(f"""
**Model**: Ridge Regression — a statistical model that predicts a number (monthly sales)
from a set of input factors. "Ridge" adds a penalty that prevents overfitting (the model
memorizing noise instead of real patterns).

**Input Factors:**
- **Month encoding** (sin/cos): Captures the circular nature of seasons (December is close to January, not far from it)
- **Seasonal low flag**: Binary flag for May-June (the observed trough period)
- **Summer flag**: Binary flag for July-August (observed peak period)
- **Q4 flag**: Binary flag for October-December (holiday season)
- **Branch maturity**: How many months the branch has been open (newer branches are still ramping up)
- **Branch identity**: Each branch has its own baseline revenue level

**Performance:**
- **Accuracy (R-squared = {fc['train_r2']:.3f})**: The model explains {accuracy_pct:.0f}% of monthly sales variation.
  A score of 1.0 would mean perfect predictions. Above 0.6 is generally useful for business planning.
- **Average Error (MAE = {fmt(fc['train_mae'])})**: On average, each monthly prediction is off by this amount.

**Limitation**: With only one year of training data, the model cannot separate one-time events
from recurring patterns. More years of data would significantly improve accuracy.
            """)

        st.subheader("Forecast-Driven Actions")
        priority_legend()
        for a in fc["actions"]:
            action_card(a)

    # ── Tab 2: Seasonal Patterns ──────────────────────────────────────────
    with tab2:
        ss = analyses["seasonal"]

        info_box(
            "Why Seasonal Patterns Matter",
            "Understanding when revenue naturally rises and falls lets you <b>plan ahead</b>: "
            "staff up before peak months, cut costs during low months, and time promotions "
            "for maximum impact. The chart below shows the 2025 monthly pattern."
        )

        st.subheader("2025 Monthly Revenue Pattern")

        chain = ss["chain_monthly"].copy()
        fig = px.bar(
            chain,
            x="month",
            y="sales",
            labels={"sales": "Total Revenue (all branches)", "month": "Month"},
            color_discrete_sequence=[C_GREEN],
        )
        # Highlight seasonal low
        fig.add_vrect(x0=3.5, x1=5.5, fillcolor=C_RED, opacity=0.1,
                       annotation_text="Seasonal Low", annotation_position="top")
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Note on the May-June trough**: The data shows a sharp revenue dip in May-June. "
            "Ramadan 2025 actually fell in March, yet March revenue appears normal. "
            "This May-June trough may reflect end-of-school-year patterns, early summer behavior, "
            "or other local factors. Regardless of cause, the pattern is real and should be planned for."
        )

        c1, c2 = st.columns(2)
        c1.metric("Strongest Month",
                   f"{ss['peak']['month']}",
                   f"Revenue: {fmt(ss['peak']['revenue'])}",
                   help="The month with the highest chain-wide revenue in 2025")
        c2.metric("Weakest Month",
                   f"{ss['trough']['month']}",
                   f"Revenue: {fmt(ss['trough']['revenue'])}",
                   delta_color="inverse",
                   help="The month with the lowest chain-wide revenue in 2025")

        # Trough resilience
        st.subheader("Branch Resilience During Seasonal Low")
        st.caption(
            "This shows what percentage of normal revenue each branch retains during the May-June low period. "
            "Higher % = more resilient. Study resilient branches to learn what they do differently."
        )
        res = ss["trough_resilience"].reset_index()
        res.columns = ["branch", "resilience_pct"]
        res = res.dropna()
        fig2 = px.bar(
            res.sort_values("resilience_pct"),
            x="resilience_pct",
            y="branch",
            orientation="h",
            labels={"resilience_pct": "% of Normal Revenue Retained", "branch": ""},
            color_discrete_sequence=[C_GOLD],
        )
        fig2.add_vline(x=50, line_dash="dash", line_color=C_RED,
                        annotation_text="50% retention")
        fig2.update_layout(**PLOTLY_LAYOUT, height=600)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Branches above 50% are relatively resilient. Those below need targeted low-season strategies.")

        st.subheader("Seasonal Strategy Actions")
        priority_legend()
        for a in ss["actions"]:
            action_card(a)


def page_data_upload():
    """Data Upload page: upload new CSVs and reprocess."""
    st.header("Upload New Data")

    st.markdown(
        "Upload updated CSV exports from the POS system to refresh the dashboard. "
        "All four files must use the same format as the originals."
    )

    # Show current data status
    processed_dir = ROOT / "data" / "processed"
    if processed_dir.exists():
        parquets = list(processed_dir.glob("*.parquet"))
        if parquets:
            import datetime
            latest = max(p.stat().st_mtime for p in parquets)
            latest_dt = datetime.datetime.fromtimestamp(latest)
            st.success(f"Current data last processed: **{latest_dt.strftime('%Y-%m-%d %H:%M')}**")
            for p in sorted(parquets):
                try:
                    row_count = len(pd.read_parquet(p))
                    st.caption(f"  {p.name}: {row_count:,} rows")
                except Exception:
                    st.caption(f"  {p.name}: (could not read)")
        else:
            st.warning("No processed data found. Upload CSVs and click Process.")
    else:
        st.warning("No processed data found. Upload CSVs and click Process.")

    st.divider()

    st.subheader("Upload CSV Files")
    st.caption("Upload all 4 files, then click 'Process Data' to refresh the dashboard.")

    f1 = st.file_uploader(
        "File 1: Monthly Sales (REP_S_00134_SMRY.csv)",
        type=["csv"], key="upload_f1",
    )
    f2 = st.file_uploader(
        "File 2: Product Profitability (rep_s_00014_SMRY.csv)",
        type=["csv"], key="upload_f2",
    )
    f3 = st.file_uploader(
        "File 3: Sales by Group (rep_s_00191_SMRY-3.csv)",
        type=["csv"], key="upload_f3",
    )
    f4 = st.file_uploader(
        "File 4: Category Summary (rep_s_00673_SMRY.csv)",
        type=["csv"], key="upload_f4",
    )

    if st.button("Process Data", type="primary", disabled=not all([f1, f2, f3, f4])):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            # Save uploaded files with expected names
            file_map = {
                "REP_S_00134_SMRY.csv": f1,
                "rep_s_00014_SMRY.csv": f2,
                "rep_s_00191_SMRY-3.csv": f3,
                "rep_s_00673_SMRY.csv": f4,
            }
            for fname, uploaded in file_map.items():
                (tmp / fname).write_bytes(uploaded.getvalue())

            with st.spinner("Processing uploaded data..."):
                try:
                    from src.cleaning import run_cleaning
                    run_cleaning(data_dir=tmp, output_dir=ROOT / "data" / "processed")
                    st.cache_data.clear()
                    st.success("Data processed successfully! Navigate to other pages to see updated results.")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    if not all([f1, f2, f3, f4]):
        st.info("Upload all 4 CSV files to enable processing.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════


def main():
    st.set_page_config(
        page_title="Stories Coffee — Revenue Action Engine",
        page_icon="☕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS — keep as st.markdown for global style injection
    st.markdown(f"""
    <style>
        .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {C_GREEN};
        }}
        [data-testid="stSidebar"] * {{
            color: {C_CREAM} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {C_GREEN};
            font-weight: 700;
        }}
        .block-container {{
            padding-top: 2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ☕ Stories Coffee")
        st.markdown("**Revenue Action Engine**")
        st.divider()
        page = st.radio(
            "Navigate",
            [
                "Executive Overview",
                "Revenue Levers",
                "Menu Engineering",
                "Branch Playbooks",
                "Forecasting & Seasonal",
                "Upload New Data",
            ],
            label_visibility="collapsed",
        )
        st.divider()

        with st.expander("Quick Guide"):
            st.markdown("""
**Executive Overview** — Start here. Top-level numbers and urgent actions.

**Revenue Levers** — Four strategies to grow revenue: food cross-sell, drink add-ons, size upgrades, delivery.

**Menu Engineering** — Which products to keep, promote, reprice, or remove.

**Branch Playbooks** — Per-branch scorecards and peer comparisons.

**Forecasting** — 2026 sales predictions and seasonal planning.

---
*All values are in relative units. Focus on ratios and comparisons, not absolute numbers.*
            """)

        st.caption("Data: 2025 full year + Jan 2026")
        st.caption("25 branches · 551 products")
        st.caption("8 analyses · 3 predictive models")

    # Upload page doesn't need analysis data
    if page == "Upload New Data":
        page_data_upload()
        return

    # Load data
    with st.spinner("Crunching the numbers across all branches..."):
        data, analyses, ml, actions_df, cards = load_all()

    # Route to page
    if page == "Executive Overview":
        page_overview(analyses, ml, actions_df, cards)
    elif page == "Revenue Levers":
        page_revenue_levers(analyses)
    elif page == "Menu Engineering":
        page_menu_engineering(analyses, ml)
    elif page == "Branch Playbooks":
        page_branch_playbooks(analyses, ml, cards)
    elif page == "Forecasting & Seasonal":
        page_forecasting(analyses, ml)


if __name__ == "__main__":
    main()
