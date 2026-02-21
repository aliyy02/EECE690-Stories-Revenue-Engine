"""
ML Models for Stories Coffee Revenue Action Engine.

1. Sales Forecasting — Ridge Regression with seasonal features
2. Branch Clustering — K-Means segmentation
3. Menu Engineering Classification — BCG quadrant with scoring
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.utils import BRANCH_REGIONS

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def _load():
    return {
        "monthly": pd.read_parquet(DATA_DIR / "monthly_sales.parquet"),
        "products": pd.read_parquet(DATA_DIR / "product_profitability.parquet"),
        "category": pd.read_parquet(DATA_DIR / "category_summary.parquet"),
    }


# =============================================================================
# Model 1: Sales Forecasting (Ridge Regression)
# =============================================================================


def sales_forecast(data=None, forecast_months=6):
    """
    Forecast monthly branch revenue using Ridge Regression.

    Features:
    - month_sin, month_cos (cyclical encoding of month)
    - branch one-hot encoding
    - is_seasonal_low flag (May/June trough period)
    - is_summer flag (Jul/Aug)
    - months_since_open (branch maturity)

    Only trains on branches with >= 6 months of data.
    Forecasts next `forecast_months` months for all branches.
    """
    if data is None:
        data = _load()

    monthly = data["monthly"]
    m25 = monthly[monthly["year"] == 2025].copy()

    # Filter to branches with enough data
    months_active = m25[m25["sales"] > 0].groupby("branch")["month_num"].count()
    eligible = months_active[months_active >= 6].index.tolist()
    m_train = m25[m25["branch"].isin(eligible) & (m25["sales"] > 0)].copy()

    # Determine when each branch first had sales (for maturity feature)
    first_month = m_train.groupby("branch")["month_num"].min().to_dict()

    # Build features
    def build_features(df):
        X = pd.DataFrame()
        X["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        X["is_seasonal_low"] = df["month_num"].isin([5, 6]).astype(int)
        X["is_summer"] = df["month_num"].isin([7, 8]).astype(int)
        X["is_q4"] = df["month_num"].isin([10, 11, 12]).astype(int)
        X["months_active"] = df.apply(
            lambda r: r["month_num"] - first_month.get(r["branch"], 1) + 1, axis=1
        )
        # Branch encoding
        branch_dummies = pd.get_dummies(df["branch"], prefix="br", dtype=int)
        X = pd.concat([X, branch_dummies], axis=1)
        return X

    X_train = build_features(m_train)
    y_train = m_train["sales"].values

    # Train Ridge model
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)

    # In-sample metrics
    y_pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)

    # Build forecast for next months (Jan-Mar 2026 = months 1,2,3)
    forecast_rows = []
    for branch in eligible:
        for m in range(1, forecast_months + 1):
            forecast_rows.append({
                "branch": branch,
                "year": 2026,
                "month_num": m,
                "sales": 0,  # placeholder
            })
    forecast_df = pd.DataFrame(forecast_rows)
    X_forecast = build_features(forecast_df)

    # Align columns (some branches may be missing in forecast)
    for col in X_train.columns:
        if col not in X_forecast.columns:
            X_forecast[col] = 0
    X_forecast = X_forecast[X_train.columns]

    y_forecast = model.predict(X_forecast)
    y_forecast = np.maximum(y_forecast, 0)  # no negative sales

    forecast_df["predicted_sales"] = y_forecast
    forecast_df["month"] = forecast_df["month_num"].map(
        {1: "January", 2: "February", 3: "March", 4: "April",
         5: "May", 6: "June", 7: "July", 8: "August",
         9: "September", 10: "October", 11: "November", 12: "December"}
    )

    # Actuals for comparison (Jan 2026 exists in data)
    m26 = monthly[(monthly["year"] == 2026) & (monthly["sales"] > 0)]

    # Feature importance
    feature_names = X_train.columns.tolist()
    non_branch_features = [f for f in feature_names if not f.startswith("br_")]
    importance = pd.Series(
        model.coef_[: len(non_branch_features)],
        index=non_branch_features,
    ).sort_values(key=abs, ascending=False)

    actions = [
        {
            "insight": f"Model R² = {r2:.3f}, MAE = {mae:,.0f}. "
                       f"Seasonal features (May-Jun trough, summer peak) are key drivers.",
            "action": "Use forecasts to pre-plan staffing and inventory. "
                      "Reduce perishable orders 30-50% for predicted low months.",
            "priority": "HIGH",
        },
    ]

    # Flag branches with declining trend
    for branch in eligible:
        b_data = m_train[m_train["branch"] == branch].sort_values("month_num")
        if len(b_data) >= 6:
            h1 = b_data[b_data["month_num"] <= 6]["sales"].mean()
            h2 = b_data[b_data["month_num"] > 6]["sales"].mean()
            if h2 < h1 * 0.85:  # 15%+ decline
                actions.append({
                    "branch": branch,
                    "insight": f"H2 revenue declined {(1 - h2/h1)*100:.0f}% vs H1",
                    "action": "Investigate cause (competition, quality, staffing). "
                              "Consider promotional campaign.",
                    "priority": "HIGH",
                })

    return {
        "model": model,
        "train_r2": r2,
        "train_mae": mae,
        "forecast": forecast_df,
        "actuals_jan26": m26,
        "feature_importance": importance,
        "eligible_branches": eligible,
        "actions": actions,
        "disclaimer": (
            "Directional only — trained on 1 year of data (2025) with in-sample evaluation. "
            "R\u00b2 is in-sample and may overstate real predictive accuracy. "
            "Use for staffing/inventory planning, not financial commitments."
        ),
    }


# =============================================================================
# Model 2: Branch Clustering (K-Means)
# =============================================================================


def branch_clustering(data=None, n_clusters=4):
    """
    Segment branches into clusters based on performance metrics.

    Features: total revenue, food share, beverage margin, food margin,
    food attachment rate, revenue per item.
    """
    if data is None:
        data = _load()

    cat = data["category"]
    cat_only = cat[cat["row_type"] == "category"].copy()
    prods = data["products"]
    prod_rows = prods[prods["row_type"] == "product"].copy()

    bev = cat_only[cat_only["category"] == "BEVERAGES"].set_index("branch")
    food = cat_only[cat_only["category"] == "FOOD"].set_index("branch")

    features = pd.DataFrame({
        "bev_revenue": bev["revenue"],
        "food_revenue": food["revenue"],
        "bev_profit_pct": bev["profit_pct"],
        "food_profit_pct": food["profit_pct"],
        "bev_qty": bev["qty"],
        "food_qty": food["qty"],
    }).fillna(0)

    features["total_revenue"] = features["bev_revenue"] + features["food_revenue"]
    features["food_share"] = features["food_revenue"] / features["total_revenue"] * 100
    features["food_attach_rate"] = features["food_qty"] / features["bev_qty"] * 100
    features["avg_ticket"] = features["total_revenue"] / (features["bev_qty"] + features["food_qty"])

    # Add service type mix from product data
    for branch in features.index:
        b_prods = prod_rows[prod_rows["branch"] == branch]
        total_rev = b_prods["revenue"].sum()
        if total_rev > 0:
            toters_rev = b_prods[b_prods["service_type"] == "Toters"]["revenue"].sum()
            table_rev = b_prods[b_prods["service_type"] == "TABLE"]["revenue"].sum()
            features.loc[branch, "toters_pct"] = toters_rev / total_rev * 100
            features.loc[branch, "table_pct"] = table_rev / total_rev * 100
        else:
            features.loc[branch, "toters_pct"] = 0
            features.loc[branch, "table_pct"] = 0

    features["region"] = features.index.map(BRANCH_REGIONS)

    # Select clustering features (normalize)
    cluster_cols = [
        "total_revenue", "food_share", "bev_profit_pct", "food_profit_pct",
        "food_attach_rate", "avg_ticket", "toters_pct", "table_pct",
    ]
    X = features[cluster_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(X_scaled)

    # Compute inertia for different k values (for elbow plot)
    inertias = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_

    # Label clusters by dominant characteristic
    cluster_profiles = features.groupby("cluster")[cluster_cols].mean()

    # Determine cluster labels based on dominant characteristic
    cluster_labels = {}
    for c in range(n_clusters):
        profile = cluster_profiles.loc[c]
        traits = []
        if profile["table_pct"] > 50:
            traits.append("Dine-In")
        if profile["toters_pct"] > 10:
            traits.append("Delivery-Heavy")
        if not traits:
            traits.append("Takeaway-Focused")

        # Revenue tier
        rev_pctile = cluster_profiles["total_revenue"].rank(pct=True)[c]
        if rev_pctile >= 0.75:
            traits.insert(0, "High-Revenue")
        elif rev_pctile <= 0.25:
            traits.insert(0, "Emerging")
        else:
            traits.insert(0, "Mid-Tier")

        cluster_labels[c] = " / ".join(traits)

    features["cluster_label"] = features["cluster"].map(cluster_labels)

    actions = []
    for c, label in cluster_labels.items():
        branches = features[features["cluster"] == c].index.tolist()
        profile = cluster_profiles.loc[c]

        if "Emerging" in label:
            action_text = ("Focus on growth: delivery onboarding, local marketing, "
                          "menu optimization for this segment")
            priority = "HIGH"
        elif "High-Revenue" in label:
            action_text = ("Protect revenue: maintain quality, optimize peak staffing, "
                          "test premium items here first")
            priority = "MEDIUM"
        elif "Dine-In" in label:
            action_text = (f"Optimize dine-in experience (table_pct={profile['table_pct']:.0f}%). "
                          "Focus on ambiance, food attachment, higher avg ticket.")
            priority = "MEDIUM"
        elif "Delivery" in label:
            action_text = (f"Leverage delivery channel ({profile['toters_pct']:.0f}% Toters). "
                          "Optimize packaging, delivery menu, promotions.")
            priority = "MEDIUM"
        else:
            action_text = (f"Apply targeted strategy: food share {profile['food_share']:.0f}%, "
                          f"avg ticket {profile['avg_ticket']:,.0f}")
            priority = "MEDIUM"

        actions.append({
            "cluster": label,
            "insight": f"Cluster '{label}': {len(branches)} branches, "
                       f"avg revenue {profile['total_revenue']:,.0f}",
            "action": action_text,
            "branches": branches,
            "priority": priority,
        })

    return {
        "features": features.reset_index(),
        "cluster_profiles": cluster_profiles,
        "cluster_labels": cluster_labels,
        "kmeans": kmeans,
        "scaler": scaler,
        "inertias": inertias,
        "actions": actions,
    }


# =============================================================================
# Model 3: Menu Engineering Classification (BCG Quadrant with Scoring)
# =============================================================================


def menu_engineering(data=None):
    """
    Enhanced BCG classification with composite scores for menu optimization.

    Scores each product on:
    - Volume score (qty percentile within category)
    - Margin score (profit margin percentile)
    - Revenue contribution score
    - Composite "menu power" score
    """
    if data is None:
        data = _load()

    prods = data["products"]
    prod_rows = prods[prods["row_type"] == "product"].copy()

    # Filter out modifiers and zero-revenue COMBO toppings
    mod_prefixes = ("ADD ", "REPLACE ", "DECAFE")
    prod_rows = prod_rows[
        ~(prod_rows["product"].str.startswith(mod_prefixes[0])
          | prod_rows["product"].str.startswith(mod_prefixes[1])
          | prod_rows["product"].str.startswith(mod_prefixes[2]))
    ].copy()
    # Remove zero-revenue COMBO toppings (yoghurt bowl add-ons)
    prod_rows = prod_rows[
        ~((prod_rows["product"].str.contains("COMBO", na=False))
          & (prod_rows["category"] == "FOOD")
          & (prod_rows["revenue"] == 0))
    ]

    # Aggregate across branches
    agg = (
        prod_rows.groupby(["product", "category", "division"])
        .agg(
            qty=("qty", "sum"),
            revenue=("revenue", "sum"),
            cost=("cost", "sum"),
            profit=("profit", "sum"),
            n_branches=("branch", "nunique"),
        )
        .reset_index()
    )
    agg["profit_margin"] = np.where(
        agg["revenue"] > 0, agg["profit"] / agg["revenue"] * 100, 0
    )
    agg["avg_price"] = np.where(agg["qty"] > 0, agg["revenue"] / agg["qty"], 0)

    # Percentile scores within category
    results = []
    for cat in ["BEVERAGES", "FOOD"]:
        cat_df = agg[agg["category"] == cat].copy()
        if cat_df.empty:
            continue

        cat_df["volume_score"] = cat_df["qty"].rank(pct=True) * 100
        cat_df["margin_score"] = cat_df["profit_margin"].rank(pct=True) * 100
        cat_df["revenue_score"] = cat_df["revenue"].rank(pct=True) * 100
        cat_df["profit_score"] = cat_df["profit"].rank(pct=True) * 100
        cat_df["breadth_score"] = cat_df["n_branches"].rank(pct=True) * 100

        # Composite menu power score (weighted)
        cat_df["menu_power"] = (
            cat_df["volume_score"] * 0.25
            + cat_df["margin_score"] * 0.25
            + cat_df["profit_score"] * 0.30
            + cat_df["breadth_score"] * 0.20
        ).round(1)

        # BCG classification
        qty_median = cat_df["qty"].median()
        margin_median = cat_df["profit_margin"].median()

        def classify(row):
            hv = row["qty"] >= qty_median
            hm = row["profit_margin"] >= margin_median
            if hv and hm:
                return "Star"
            elif hv and not hm:
                return "Plow Horse"
            elif not hv and hm:
                return "Puzzle"
            else:
                return "Dog"

        cat_df["bcg_class"] = cat_df.apply(classify, axis=1)
        results.append(cat_df)

    menu = pd.concat(results, ignore_index=True)
    menu = menu.sort_values("menu_power", ascending=False)

    # Top recommendations
    stars = menu[menu["bcg_class"] == "Star"].sort_values("menu_power", ascending=False)
    puzzles = menu[menu["bcg_class"] == "Puzzle"].sort_values("menu_power", ascending=False)
    dogs = menu[menu["bcg_class"] == "Dog"].sort_values("menu_power")
    plow = menu[menu["bcg_class"] == "Plow Horse"].sort_values("qty", ascending=False)

    # Summary stats
    summary = menu.groupby("bcg_class").agg(
        count=("product", "count"),
        total_qty=("qty", "sum"),
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
        avg_margin=("profit_margin", "mean"),
        avg_menu_power=("menu_power", "mean"),
    ).round(1)

    actions = []
    # Stars to protect
    for _, row in stars.head(5).iterrows():
        actions.append({
            "product": row["product"],
            "insight": f"Star: Menu Power {row['menu_power']:.0f}, "
                       f"{row['qty']:,.0f} units, {row['profit_margin']:.0f}% margin, "
                       f"sold at {row['n_branches']} branches",
            "action": "PROTECT: Never discount. Feature on menu boards, "
                      "social media, staff favorites.",
            "category": "PROTECT",
            "priority": "HIGH",
        })

    # High-power puzzles (hidden gems)
    for _, row in puzzles.head(5).iterrows():
        actions.append({
            "product": row["product"],
            "insight": f"Hidden Gem: {row['profit_margin']:.0f}% margin, "
                       f"Menu Power {row['menu_power']:.0f}, only {row['qty']:,.0f} units",
            "action": "PROMOTE: Better menu placement, staff recommendation scripts, "
                      "free tastings, social media feature.",
            "category": "PROMOTE",
            "priority": "HIGH",
        })

    # Dogs to kill (lowest menu power)
    for _, row in dogs.head(5).iterrows():
        actions.append({
            "product": row["product"],
            "insight": f"Remove candidate: Menu Power {row['menu_power']:.0f}, "
                       f"{row['qty']:,.0f} units, {row['profit_margin']:.0f}% margin",
            "action": "REMOVE from menu. Simplifies operations, reduces waste.",
            "category": "KILL",
            "priority": "MEDIUM",
        })

    # Plow horses to reprice
    for _, row in plow.head(3).iterrows():
        potential = row["qty"] * row["avg_price"] * 0.05  # 5% price increase
        actions.append({
            "product": row["product"],
            "insight": f"Plow Horse: {row['qty']:,.0f} units but {row['profit_margin']:.0f}% margin "
                       f"(avg price {row['avg_price']:,.0f})",
            "action": f"REPRICE +5%: Potential additional revenue ~{potential:,.0f}. "
                       "High volume = inelastic demand.",
            "category": "REPRICE",
            "priority": "MEDIUM",
        })

    return {
        "menu": menu,
        "summary": summary,
        "stars": stars,
        "puzzles": puzzles,
        "dogs": dogs,
        "plow_horses": plow,
        "actions": actions,
    }


# =============================================================================
# Run All Models
# =============================================================================


def run_all_models():
    """Run all 3 ML models and return results."""
    data = _load()
    print("Running ML models...")

    results = {}

    print("  [1/3] Sales Forecasting (Ridge Regression)...")
    results["forecast"] = sales_forecast(data)
    r = results["forecast"]
    print(f"        → R² = {r['train_r2']:.3f}, MAE = {r['train_mae']:,.0f}")
    print(f"        → {len(r['eligible_branches'])} branches forecasted, "
          f"{len(r['actions'])} actions")

    print("  [2/3] Branch Clustering (K-Means)...")
    results["clusters"] = branch_clustering(data)
    c = results["clusters"]
    print(f"        → {len(c['cluster_labels'])} clusters identified")
    for cid, label in c["cluster_labels"].items():
        n = len(c["features"][c["features"]["cluster"] == cid])
        print(f"          Cluster {cid}: {label} ({n} branches)")

    print("  [3/3] Menu Engineering Classification...")
    results["menu"] = menu_engineering(data)
    m = results["menu"]
    print(f"        → {len(m['menu'])} products classified")
    print(f"        → {len(m['actions'])} actions generated")
    print(f"        Summary:")
    print(m["summary"][["count", "total_profit", "avg_margin"]].to_string())

    total_actions = sum(len(r.get("actions", [])) for r in results.values())
    print(f"\nTotal ML-driven actions: {total_actions}")

    return results
