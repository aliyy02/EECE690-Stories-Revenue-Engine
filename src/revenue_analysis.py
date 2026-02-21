"""
Revenue Analysis Engine for Stories Coffee.

8 analyses, each returning a dict with:
- data: DataFrame(s) of results
- actions: list of (insight, action) tuples
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import BRANCH_REGIONS

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Brand colors
C_GREEN = "#1B3A2D"
C_GOLD = "#C8A96E"
C_CREAM = "#F5F0EB"
C_RED = "#C0392B"
C_BLUE = "#2980B9"


def _load():
    """Load all parquet files."""
    return {
        "monthly": pd.read_parquet(DATA_DIR / "monthly_sales.parquet"),
        "products": pd.read_parquet(DATA_DIR / "product_profitability.parquet"),
        "groups": pd.read_parquet(DATA_DIR / "sales_by_group.parquet"),
        "category": pd.read_parquet(DATA_DIR / "category_summary.parquet"),
    }


def _product_rows(df):
    return df[df["row_type"] == "product"].copy()


def _is_modifier(product_name):
    """Check if a product is a modifier/add-on (not a standalone item)."""
    mod_prefixes = ("ADD ", "REPLACE ", "DECAFE")
    return product_name.startswith(mod_prefixes)


def _is_food_topping(row):
    """Check if a food item is a zero-revenue COMBO topping (yoghurt bowl add-on)."""
    return (
        "COMBO" in str(row.get("product", ""))
        and row.get("category") == "FOOD"
        and row.get("revenue", 1) == 0
    )


def _extract_size(product_name):
    """Extract (base_product, size) from a product name."""
    for s in ("LARGE", "MEDIUM", "SMALL"):
        if product_name.endswith(f" {s}"):
            return product_name[: -len(s) - 1].strip(), s
    return product_name, None


# =============================================================================
# Analysis 1: Cross-Sell Opportunity Matrix
# =============================================================================


def cross_sell_matrix(data=None):
    """
    Food attachment rate by branch: food qty / non-modifier beverage qty.
    Branches with low attachment rate are cross-sell opportunities.
    """
    if data is None:
        data = _load()

    # Use product-level data to get non-modifier beverage qty (actual drink orders)
    # and exclude zero-revenue COMBO toppings from food (yoghurt bowl add-ons)
    prods = _product_rows(data["products"])
    prods["is_modifier"] = prods["product"].apply(_is_modifier)
    prods["is_food_topping"] = prods.apply(_is_food_topping, axis=1)
    real_bev = prods[(~prods["is_modifier"]) & (prods["category"] == "BEVERAGES")]
    food_prods = prods[(prods["category"] == "FOOD") & (~prods["is_food_topping"])]

    bev_by_branch = real_bev.groupby("branch").agg(
        bev_qty=("qty", "sum"), bev_revenue=("revenue", "sum"), bev_profit=("profit", "sum")
    )
    food_by_branch = food_prods.groupby("branch").agg(
        food_qty=("qty", "sum"), food_revenue=("revenue", "sum"), food_profit=("profit", "sum")
    )

    matrix = bev_by_branch.join(food_by_branch, how="outer").fillna(0)

    matrix["food_attach_rate"] = (matrix["food_qty"] / matrix["bev_qty"] * 100).round(1)
    matrix["food_revenue_share"] = (
        matrix["food_revenue"] / (matrix["bev_revenue"] + matrix["food_revenue"]) * 100
    ).round(1)
    matrix["region"] = matrix.index.map(BRANCH_REGIONS)
    matrix = matrix.sort_values("food_attach_rate", ascending=False).reset_index()

    avg_attach = matrix["food_attach_rate"].mean()

    # Identify low-attachment branches (below average)
    low = matrix[matrix["food_attach_rate"] < avg_attach].sort_values("food_attach_rate")
    high = matrix[matrix["food_attach_rate"] >= avg_attach].sort_values(
        "food_attach_rate", ascending=False
    )

    actions = []
    for _, row in low.iterrows():
        gap = avg_attach - row["food_attach_rate"]
        potential_qty = row["bev_qty"] * gap / 100
        actions.append(
            {
                "branch": row["branch"],
                "insight": f"Food attach rate {row['food_attach_rate']:.0f}% "
                f"(vs {avg_attach:.0f}% avg) — {gap:.0f}pp below average",
                "action": f"Train staff on food pairing scripts. "
                f"Potential: ~{potential_qty:,.0f} additional food items/year",
                "priority": "HIGH" if gap > 15 else "MEDIUM",
            }
        )

    return {
        "data": matrix,
        "avg_attach_rate": avg_attach,
        "low_attach_branches": low,
        "high_attach_branches": high,
        "actions": actions,
    }


# =============================================================================
# Analysis 2: Modifier Upsell Engine
# =============================================================================


def modifier_upsell(data=None):
    """
    Modifier attach rate by branch and product.
    Identifies branches under-selling modifiers.
    """
    if data is None:
        data = _load()
    prods = _product_rows(data["products"])

    # Separate modifiers vs real products
    prods["is_modifier"] = prods["product"].apply(_is_modifier)
    mods = prods[prods["is_modifier"]]
    real = prods[~prods["is_modifier"]]

    # Only beverage modifiers (modifiers are in BEVERAGES category)
    bev_real = real[real["category"] == "BEVERAGES"]

    # Branch-level modifier stats
    mod_by_branch = mods.groupby("branch").agg(
        mod_qty=("qty", "sum"), mod_revenue=("revenue", "sum")
    )
    bev_by_branch = bev_real.groupby("branch")["qty"].sum().rename("bev_qty")
    branch_stats = mod_by_branch.join(bev_by_branch, how="outer").fillna(0)
    branch_stats["mod_attach_rate"] = (branch_stats["mod_qty"] / branch_stats["bev_qty"] * 100).round(1)
    branch_stats = branch_stats.sort_values("mod_attach_rate", ascending=False).reset_index()

    # Top modifiers chain-wide
    top_mods = (
        mods.groupby("product")
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("qty", ascending=False)
    )

    # Revenue-generating modifiers (ADD SHOT, ADD SYRUP, etc.) vs free combo toppings
    paid_mods = top_mods[top_mods["revenue"] > 0]
    free_mods = top_mods[top_mods["revenue"] == 0]

    avg_attach = branch_stats["mod_attach_rate"].mean()

    actions = []
    low = branch_stats[branch_stats["mod_attach_rate"] < avg_attach]
    for _, row in low.iterrows():
        gap = avg_attach - row["mod_attach_rate"]
        actions.append(
            {
                "branch": row["branch"],
                "insight": f"Modifier attach rate {row['mod_attach_rate']:.0f}% "
                f"(vs {avg_attach:.0f}% avg)",
                "action": "Train staff: 'Would you like an extra shot or oat milk?' "
                f"at POS. Gap = {gap:.0f}pp",
                "priority": "HIGH" if gap > 20 else "MEDIUM",
            }
        )

    return {
        "branch_stats": branch_stats,
        "top_paid_modifiers": paid_mods.head(15),
        "top_free_modifiers": free_mods.head(15),
        "avg_attach_rate": avg_attach,
        "actions": actions,
    }


# =============================================================================
# Analysis 3: Menu Kill List (BCG Matrix)
# =============================================================================


def menu_bcg_matrix(data=None):
    """
    Classify products into Stars/Plow Horses/Puzzles/Dogs using
    volume (qty) and margin (profit_pct) relative to category median.
    """
    if data is None:
        data = _load()
    prods = _product_rows(data["products"])

    # Filter to real products only (no modifiers)
    prods = prods[~prods["product"].apply(_is_modifier)].copy()

    # Aggregate across branches
    agg = (
        prods.groupby(["product", "category", "division"])
        .agg(
            qty=("qty", "sum"),
            revenue=("revenue", "sum"),
            cost=("cost", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    agg["profit_margin"] = np.where(agg["revenue"] > 0, agg["profit"] / agg["revenue"] * 100, 0)

    # BCG classification per category
    results = []
    for cat in ["BEVERAGES", "FOOD"]:
        cat_df = agg[agg["category"] == cat].copy()
        if cat_df.empty:
            continue

        qty_median = cat_df["qty"].median()
        margin_median = cat_df["profit_margin"].median()

        def classify(row):
            high_vol = row["qty"] >= qty_median
            high_margin = row["profit_margin"] >= margin_median
            if high_vol and high_margin:
                return "Star"
            elif high_vol and not high_margin:
                return "Plow Horse"
            elif not high_vol and high_margin:
                return "Puzzle"
            else:
                return "Dog"

        cat_df["bcg_class"] = cat_df.apply(classify, axis=1)
        cat_df["qty_median"] = qty_median
        cat_df["margin_median"] = margin_median
        results.append(cat_df)

    bcg = pd.concat(results, ignore_index=True)

    # Kill list: Dogs with < 25th percentile qty AND < 25th percentile margin
    dogs = bcg[bcg["bcg_class"] == "Dog"].copy()
    q25_qty = bcg["qty"].quantile(0.25)
    q25_margin = bcg["profit_margin"].quantile(0.25)
    kill_list = dogs[(dogs["qty"] < q25_qty) & (dogs["profit_margin"] < q25_margin)]
    kill_list = kill_list.sort_values("profit", ascending=True)

    # Stars to protect
    stars = bcg[bcg["bcg_class"] == "Star"].sort_values("profit", ascending=False)

    # Puzzles to promote
    puzzles = bcg[bcg["bcg_class"] == "Puzzle"].sort_values("profit_margin", ascending=False)

    # Plow horses to reprice
    plow = bcg[bcg["bcg_class"] == "Plow Horse"].sort_values("qty", ascending=False)

    actions = []
    # Kill recommendations
    for _, row in kill_list.head(10).iterrows():
        actions.append(
            {
                "product": row["product"],
                "insight": f"Dog: {row['qty']:,.0f} units, {row['profit_margin']:.1f}% margin, "
                f"${row['profit']:,.0f} profit",
                "action": f"REMOVE from menu. Saves prep time, simplifies training, "
                f"reduces waste.",
                "category": "KILL",
                "priority": "HIGH",
            }
        )

    # Puzzle promotions
    for _, row in puzzles.head(10).iterrows():
        actions.append(
            {
                "product": row["product"],
                "insight": f"Puzzle: {row['profit_margin']:.1f}% margin but only "
                f"{row['qty']:,.0f} units",
                "action": "PROMOTE: Better menu placement, staff recommendation scripts, "
                "social media feature",
                "category": "PROMOTE",
                "priority": "HIGH",
            }
        )

    # Plow horse repricing
    for _, row in plow.head(5).iterrows():
        actions.append(
            {
                "product": row["product"],
                "insight": f"Plow Horse: {row['qty']:,.0f} units but only "
                f"{row['profit_margin']:.1f}% margin",
                "action": "REPRICE: Raise price 5-8% (demand is inelastic for popular items) "
                "OR reduce COGS",
                "category": "REPRICE",
                "priority": "MEDIUM",
            }
        )

    return {
        "bcg": bcg,
        "stars": stars,
        "plow_horses": plow,
        "puzzles": puzzles,
        "dogs": dogs,
        "kill_list": kill_list,
        "actions": actions,
    }


# =============================================================================
# Analysis 4: Bundle/Combo Builder
# =============================================================================


def bundle_builder(data=None):
    """
    Identify best bundle candidates by pairing:
    - Top-volume beverage + highest-margin food in same branch/service_type
    """
    if data is None:
        data = _load()
    prods = _product_rows(data["products"])
    prods = prods[~prods["product"].apply(_is_modifier)].copy()

    # Chain-wide top beverages by qty
    bev = prods[prods["category"] == "BEVERAGES"]
    food = prods[prods["category"] == "FOOD"]

    top_bev = (
        bev.groupby("product")
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("qty", ascending=False)
        .head(10)
    )
    top_bev["margin"] = (top_bev["profit"] / top_bev["revenue"] * 100).round(1)

    # Top food by profit margin (min 1000 qty to filter out rarities)
    food_agg = (
        food.groupby("product")
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("profit", ascending=False)
    )
    food_agg["margin"] = np.where(
        food_agg["revenue"] > 0, (food_agg["profit"] / food_agg["revenue"] * 100).round(1), 0
    )
    top_food = food_agg[food_agg["qty"] > 1000].head(10)

    # Build bundle suggestions: pair each top bev with best margin food
    bundles = []
    for bev_name, bev_row in top_bev.iterrows():
        for food_name, food_row in top_food.head(3).iterrows():
            combined_price = bev_row["revenue"] / bev_row["qty"] + food_row["revenue"] / food_row["qty"]
            combined_cost = (
                (bev_row["revenue"] - bev_row["profit"]) / bev_row["qty"]
                + (food_row["revenue"] - food_row["profit"]) / food_row["qty"]
            )
            bundle_price = combined_price * 0.92  # 8% discount
            bundle_profit = bundle_price - combined_cost
            bundle_margin = bundle_profit / bundle_price * 100

            bundles.append(
                {
                    "beverage": bev_name,
                    "food": food_name,
                    "individual_price": combined_price,
                    "bundle_price": bundle_price,
                    "discount_pct": 8,
                    "bundle_margin": round(bundle_margin, 1),
                    "bev_qty": bev_row["qty"],
                }
            )

    bundles_df = pd.DataFrame(bundles)
    # Keep top bundles by margin
    bundles_df = bundles_df.sort_values("bundle_margin", ascending=False)

    actions = []
    for _, row in bundles_df.head(5).iterrows():
        actions.append(
            {
                "bundle": f"{row['beverage']} + {row['food']}",
                "insight": f"Combined margin {row['bundle_margin']:.0f}% at 8% discount. "
                f"Bev sells {row['bev_qty']:,.0f} units/year.",
                "action": f"Launch as menu board combo at ~{row['bundle_price']:,.0f} "
                f"(save ~{row['individual_price'] - row['bundle_price']:,.0f}). "
                f"Feature on counter display.",
                "priority": "HIGH",
            }
        )

    return {
        "top_beverages": top_bev,
        "top_foods": top_food,
        "bundles": bundles_df,
        "actions": actions,
    }


# =============================================================================
# Analysis 5: Branch-Specific Playbooks
# =============================================================================


def branch_playbooks(data=None):
    """
    Per-branch strengths and weaknesses relative to chain averages.
    """
    if data is None:
        data = _load()

    cat = data["category"]
    cat = cat[cat["row_type"] == "category"].copy()
    prods = _product_rows(data["products"])
    prods_real = prods[~prods["product"].apply(_is_modifier)].copy()
    monthly = data["monthly"]
    monthly_2025 = monthly[monthly["year"] == 2025]

    playbooks = {}

    # Per-branch metrics
    for branch in sorted(cat["branch"].unique()):
        if branch == "Unknown/Closed":
            continue

        b_cat = cat[cat["branch"] == branch]
        b_prods = prods_real[prods_real["branch"] == branch]
        b_monthly = monthly_2025[monthly_2025["branch"] == branch]
        b_mods = prods[
            (prods["branch"] == branch) & (prods["product"].apply(_is_modifier))
        ]

        bev = b_cat[b_cat["category"] == "BEVERAGES"]
        food = b_cat[b_cat["category"] == "FOOD"]

        total_rev = b_cat["revenue"].sum()
        total_profit = b_cat["profit"].sum()
        # Use non-modifier beverage qty as denominator (actual drink orders)
        b_real_bev = b_prods[
            (~b_prods["product"].apply(_is_modifier)) & (b_prods["category"] == "BEVERAGES")
        ]
        bev_qty = b_real_bev["qty"].sum() if len(b_real_bev) > 0 else 0
        # Exclude zero-revenue COMBO toppings from food qty
        b_all_prods = prods[prods["branch"] == branch]
        b_real_food = b_all_prods[
            (b_all_prods["category"] == "FOOD") & (~b_all_prods.apply(_is_food_topping, axis=1))
        ]
        food_qty = b_real_food["qty"].sum() if len(b_real_food) > 0 else 0
        food_attach = (food_qty / bev_qty * 100) if bev_qty > 0 else 0

        # Service type mix
        svc = b_prods.groupby("service_type")["revenue"].sum()
        toters_pct = svc.get("Toters", 0) / total_rev * 100 if total_rev > 0 else 0
        table_pct = svc.get("TABLE", 0) / total_rev * 100 if total_rev > 0 else 0

        # Top products
        top_prods = (
            b_prods.groupby("product")["profit"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        # Modifier attach (modifiers per 100 non-modifier drinks)
        mod_qty = b_mods["qty"].sum()
        mod_attach = (mod_qty / bev_qty * 100) if bev_qty > 0 else 0  # bev_qty is already non-modifier

        # Seasonality - peak/trough months
        if not b_monthly.empty:
            peak_month = b_monthly.loc[b_monthly["sales"].idxmax(), "month"]
            trough_month = b_monthly.loc[
                b_monthly[b_monthly["sales"] > 0]["sales"].idxmin(), "month"
            ] if (b_monthly["sales"] > 0).any() else "N/A"
        else:
            peak_month = "N/A"
            trough_month = "N/A"

        profit_margin = (total_profit / total_rev * 100) if total_rev > 0 else 0

        playbooks[branch] = {
            "region": BRANCH_REGIONS.get(branch, "Unknown"),
            "total_revenue": total_rev,
            "total_profit": total_profit,
            "profit_margin": round(profit_margin, 1),
            "food_attach_rate": round(food_attach, 1),
            "mod_attach_rate": round(mod_attach, 1),
            "toters_pct": round(toters_pct, 1),
            "table_pct": round(table_pct, 1),
            "peak_month": peak_month,
            "trough_month": trough_month,
            "top_products": top_prods.to_dict(),
        }

    # Build summary DataFrame
    summary = pd.DataFrame(playbooks).T
    summary.index.name = "branch"
    summary = summary.reset_index()

    # Chain averages for comparison
    avgs = {
        "food_attach_rate": summary["food_attach_rate"].mean(),
        "mod_attach_rate": summary["mod_attach_rate"].mean(),
        "toters_pct": summary["toters_pct"].mean(),
        "profit_margin": summary["profit_margin"].mean(),
    }

    # Generate per-branch actions (each action as separate entry with insight/action keys)
    actions = []
    for _, row in summary.iterrows():
        if row["food_attach_rate"] < avgs["food_attach_rate"] - 5:
            actions.append({
                "branch": row["branch"],
                "insight": f"Food attach {row['food_attach_rate']:.0f}% vs "
                           f"{avgs['food_attach_rate']:.0f}% chain avg",
                "action": "Train staff on food pairing scripts (morning: pastry, "
                          "afternoon: cookie/donut)",
                "priority": "HIGH" if row["food_attach_rate"] < avgs["food_attach_rate"] - 15 else "MEDIUM",
            })
        if row["mod_attach_rate"] < avgs["mod_attach_rate"] - 10:
            actions.append({
                "branch": row["branch"],
                "insight": f"Modifier rate {row['mod_attach_rate']:.0f}% vs "
                           f"{avgs['mod_attach_rate']:.0f}% chain avg",
                "action": "Implement upsell prompts at POS: 'Add extra shot? Oat milk?'",
                "priority": "HIGH" if row["mod_attach_rate"] < avgs["mod_attach_rate"] - 20 else "MEDIUM",
            })
        if row["toters_pct"] < 1 and avgs["toters_pct"] > 2:
            actions.append({
                "branch": row["branch"],
                "insight": f"No Toters delivery presence ({row['toters_pct']:.1f}%)",
                "action": "Onboard to Toters delivery platform for incremental revenue",
                "priority": "MEDIUM",
            })
        if row["profit_margin"] < avgs["profit_margin"] - 3:
            actions.append({
                "branch": row["branch"],
                "insight": f"Profit margin {row['profit_margin']:.0f}% vs "
                           f"{avgs['profit_margin']:.0f}% chain avg",
                "action": "Audit COGS and waste at this branch. Check for theft/spillage.",
                "priority": "HIGH",
            })

    return {
        "playbooks": playbooks,
        "summary": summary,
        "chain_averages": avgs,
        "actions": actions,
    }


# =============================================================================
# Analysis 6: Seasonal Revenue Strategy
# =============================================================================


def seasonal_strategy(data=None):
    """
    Monthly revenue patterns, Ramadan impact, and seasonal recommendations.
    """
    if data is None:
        data = _load()

    monthly = data["monthly"]
    m25 = monthly[monthly["year"] == 2025].copy()

    # Chain-wide monthly totals
    chain_monthly = m25.groupby(["month", "month_num"])["sales"].sum().reset_index()
    chain_monthly = chain_monthly.sort_values("month_num")

    peak = chain_monthly.loc[chain_monthly["sales"].idxmax()]
    trough = chain_monthly.loc[chain_monthly[chain_monthly["sales"] > 0]["sales"].idxmin()]

    # Seasonal low period (May-June) — data shows trough here regardless of Ramadan timing
    trough_rev = chain_monthly[chain_monthly["month_num"].isin([5, 6])]["sales"].sum()
    summer_rev = chain_monthly[chain_monthly["month_num"].isin([7, 8])]["sales"].sum()
    avg_monthly = chain_monthly["sales"].mean()

    # Branch resilience during seasonal low
    branch_trough = m25[m25["month_num"].isin([5, 6])].groupby("branch")["sales"].sum()
    branch_avg = m25[~m25["month_num"].isin([5, 6])].groupby("branch")["sales"].mean() * 2
    trough_resilience = (branch_trough / branch_avg * 100).sort_values(ascending=False)

    # Q4 holiday lift
    q4_rev = chain_monthly[chain_monthly["month_num"].isin([10, 11, 12])]["sales"].sum()
    q3_rev = chain_monthly[chain_monthly["month_num"].isin([7, 8, 9])]["sales"].sum()

    actions = [
        {
            "season": "Seasonal Low (May-Jun)",
            "insight": f"Revenue drops to {trough_rev/1e6:.0f}M "
            f"({trough_rev/avg_monthly/2*100:.0f}% of avg). "
            f"Trough month: {trough['month']}",
            "action": "COST PLAY: Cut staff hours 30%, reduce perishable orders 50%, "
            "launch value-bundle specials to maintain traffic",
            "priority": "HIGH",
        },
        {
            "season": "Summer Peak (Jul-Aug)",
            "insight": f"Revenue peaks at {summer_rev/1e6:.0f}M "
            f"({summer_rev/avg_monthly/2*100:.0f}% of avg). "
            f"Peak month: {peak['month']}",
            "action": "REVENUE PLAY: Hire seasonal staff, max cold bar inventory, "
            "push Frapp promotions on social media",
            "priority": "HIGH",
        },
        {
            "season": "Q4 Holidays (Oct-Dec)",
            "insight": f"Q4 revenue {q4_rev/1e6:.0f}M vs Q3 {q3_rev/1e6:.0f}M",
            "action": "Launch seasonal menu (holiday drinks), gift card promotions, "
            "mall branch extended hours",
            "priority": "MEDIUM",
        },
    ]

    # Resilient branches during seasonal low
    top_resilient = trough_resilience.head(5)
    for branch, pct in top_resilient.items():
        if pct > 60:
            actions.append(
                {
                    "season": "Seasonal Low",
                    "insight": f"{branch} retains {pct:.0f}% of normal revenue during seasonal low",
                    "action": f"Study {branch}'s low-season playbook and replicate across "
                    f"low-resilience branches",
                    "priority": "MEDIUM",
                }
            )

    return {
        "chain_monthly": chain_monthly,
        "peak": {"month": peak["month"], "revenue": peak["sales"]},
        "trough": {"month": trough["month"], "revenue": trough["sales"]},
        "trough_resilience": trough_resilience,
        "actions": actions,
    }


# =============================================================================
# Analysis 7: Channel Expansion (Toters/Delivery)
# =============================================================================


def channel_expansion(data=None):
    """
    Toters penetration by branch. Identifies branches with no/low delivery presence.
    """
    if data is None:
        data = _load()

    prods = _product_rows(data["products"])
    prods = prods[~prods["product"].apply(_is_modifier)].copy()

    # Revenue by branch and service type
    svc = (
        prods.groupby(["branch", "service_type"])
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        .reset_index()
    )

    # Pivot to get columns per service type
    pivot = svc.pivot_table(
        index="branch",
        columns="service_type",
        values=["revenue", "qty"],
        fill_value=0,
    )
    pivot.columns = [f"{col[1].lower()}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    # Calculate totals and percentages
    rev_cols = [c for c in pivot.columns if c.endswith("_revenue")]
    pivot["total_revenue"] = pivot[rev_cols].sum(axis=1)
    if "toters_revenue" in pivot.columns:
        pivot["toters_pct"] = (pivot["toters_revenue"] / pivot["total_revenue"] * 100).round(1)
    else:
        pivot["toters_pct"] = 0.0
        pivot["toters_revenue"] = 0.0

    if "table_revenue" in pivot.columns:
        pivot["table_pct"] = (pivot["table_revenue"] / pivot["total_revenue"] * 100).round(1)
    else:
        pivot["table_pct"] = 0.0

    pivot["region"] = pivot["branch"].map(BRANCH_REGIONS)
    pivot = pivot.sort_values("toters_pct", ascending=False)

    # Branches with zero or minimal Toters
    no_toters = pivot[pivot["toters_pct"] < 1].sort_values("total_revenue", ascending=False)
    has_toters = pivot[pivot["toters_pct"] >= 1].sort_values("toters_pct", ascending=False)

    avg_toters_pct = has_toters["toters_pct"].mean() if len(has_toters) > 0 else 0

    actions = []
    for _, row in no_toters.head(10).iterrows():
        potential = row["total_revenue"] * avg_toters_pct / 100
        actions.append(
            {
                "branch": row["branch"],
                "insight": f"Toters = {row['toters_pct']:.1f}% of revenue "
                f"(branch does {row['total_revenue']:,.0f} total)",
                "action": f"Onboard to Toters delivery. Potential incremental revenue: "
                f"~{potential:,.0f}/year based on {avg_toters_pct:.0f}% avg penetration",
                "priority": "HIGH" if row["total_revenue"] > 30_000_000 else "MEDIUM",
            }
        )

    return {
        "channel_mix": pivot,
        "no_toters": no_toters,
        "has_toters": has_toters,
        "avg_toters_pct": avg_toters_pct,
        "actions": actions,
    }


# =============================================================================
# Analysis 8: Size Upsell Opportunity
# =============================================================================


def size_upsell(data=None):
    """
    Size distribution across products. Identifies upsell potential from Small→Medium→Large.
    """
    if data is None:
        data = _load()

    prods = _product_rows(data["products"])
    prods = prods[~prods["product"].apply(_is_modifier)].copy()

    # Extract size info
    prods["base_product"] = prods["product"].apply(lambda x: _extract_size(x)[0])
    prods["size"] = prods["product"].apply(lambda x: _extract_size(x)[1])

    sized = prods[prods["size"].notna()].copy()

    # Chain-wide size distribution
    size_dist = sized.groupby("size").agg(
        qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum")
    )
    size_dist["avg_price"] = size_dist["revenue"] / size_dist["qty"]
    size_dist["avg_profit"] = size_dist["profit"] / size_dist["qty"]
    size_dist["pct_of_total"] = (size_dist["qty"] / size_dist["qty"].sum() * 100).round(1)

    # Size distribution by branch
    branch_size = (
        sized.groupby(["branch", "size"])["qty"]
        .sum()
        .unstack(fill_value=0)
    )
    for s in ["SMALL", "MEDIUM", "LARGE"]:
        if s not in branch_size.columns:
            branch_size[s] = 0
    branch_size["total"] = branch_size.sum(axis=1)
    branch_size["large_pct"] = (branch_size["LARGE"] / branch_size["total"] * 100).round(1)
    branch_size["small_pct"] = (branch_size["SMALL"] / branch_size["total"] * 100).round(1)
    branch_size = branch_size.sort_values("large_pct", ascending=False).reset_index()

    # Per-product size analysis (products that have all 3 sizes)
    product_sizes = (
        sized.groupby(["base_product", "size"])
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    products_with_3 = (
        product_sizes.groupby("base_product")["size"]
        .nunique()
        .loc[lambda x: x == 3]
        .index
    )
    three_size = product_sizes[product_sizes["base_product"].isin(products_with_3)].copy()
    three_size["avg_price"] = three_size["revenue"] / three_size["qty"]

    # Calculate upsell revenue potential
    # If 10% of Small orders upgrade to Medium, and 10% of Medium upgrade to Large
    if "SMALL" in size_dist.index and "MEDIUM" in size_dist.index:
        small_qty = size_dist.loc["SMALL", "qty"]
        med_qty = size_dist.loc["MEDIUM", "qty"]
        small_price = size_dist.loc["SMALL", "avg_price"]
        med_price = size_dist.loc["MEDIUM", "avg_price"]
        large_price = size_dist.loc["LARGE", "avg_price"] if "LARGE" in size_dist.index else med_price * 1.2

        upgrade_10pct_revenue = (
            small_qty * 0.1 * (med_price - small_price)
            + med_qty * 0.1 * (large_price - med_price)
        )
    else:
        upgrade_10pct_revenue = 0

    avg_large_pct = branch_size["large_pct"].mean()

    actions = [
        {
            "insight": f"Size distribution: Small {size_dist.loc['SMALL', 'pct_of_total']:.0f}%, "
            f"Medium {size_dist.loc['MEDIUM', 'pct_of_total']:.0f}%, "
            f"Large {size_dist.loc['LARGE', 'pct_of_total']:.0f}%"
            if "LARGE" in size_dist.index
            else "Size data incomplete",
            "action": f"If 10% of Small/Medium upgrade → +{upgrade_10pct_revenue:,.0f} revenue. "
            f"Train staff: 'Would you like a Medium? Only [X] more.'",
            "priority": "HIGH",
        }
    ]

    # Branches with lowest large_pct (most room for upsell)
    low_large = branch_size[branch_size["large_pct"] < avg_large_pct].sort_values("large_pct")
    for _, row in low_large.head(5).iterrows():
        actions.append(
            {
                "branch": row["branch"],
                "insight": f"Large = {row['large_pct']:.0f}% (vs {avg_large_pct:.0f}% avg)",
                "action": "Prioritize size upsell training at this branch. "
                "Use decoy pricing on menu boards.",
                "priority": "MEDIUM",
            }
        )

    return {
        "size_dist": size_dist,
        "branch_size": branch_size,
        "three_size_products": three_size,
        "upsell_10pct_revenue": upgrade_10pct_revenue,
        "actions": actions,
    }


# =============================================================================
# Run All Analyses
# =============================================================================


def run_all_analyses():
    """Run all 8 analyses and return results dict."""
    data = _load()
    print("Running all 8 revenue analyses...")

    results = {}

    print("  [1/8] Cross-Sell Opportunity Matrix...")
    results["cross_sell"] = cross_sell_matrix(data)
    print(f"        → {len(results['cross_sell']['actions'])} actions generated")

    print("  [2/8] Modifier Upsell Engine...")
    results["modifier_upsell"] = modifier_upsell(data)
    print(f"        → {len(results['modifier_upsell']['actions'])} actions generated")

    print("  [3/8] Menu BCG Matrix...")
    results["bcg"] = menu_bcg_matrix(data)
    print(f"        → {len(results['bcg']['actions'])} actions generated")

    print("  [4/8] Bundle/Combo Builder...")
    results["bundles"] = bundle_builder(data)
    print(f"        → {len(results['bundles']['actions'])} actions generated")

    print("  [5/8] Branch Playbooks...")
    results["playbooks"] = branch_playbooks(data)
    print(f"        → {len(results['playbooks']['actions'])} actions generated")

    print("  [6/8] Seasonal Strategy...")
    results["seasonal"] = seasonal_strategy(data)
    print(f"        → {len(results['seasonal']['actions'])} actions generated")

    print("  [7/8] Channel Expansion (Toters)...")
    results["channels"] = channel_expansion(data)
    print(f"        → {len(results['channels']['actions'])} actions generated")

    print("  [8/8] Size Upsell Opportunity...")
    results["size_upsell"] = size_upsell(data)
    print(f"        → {len(results['size_upsell']['actions'])} actions generated")

    total_actions = sum(len(r.get("actions", [])) for r in results.values())
    print(f"\nTotal: {total_actions} actionable recommendations generated.")

    return results
