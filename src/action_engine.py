"""
Action Engine: Converts analysis results into structured, prioritized recommendations.

Every insight is paired with a specific, implementable action.
Output formats: summary DataFrame, branch playbook cards, executive brief.
"""

import pandas as pd

from src.utils import BRANCH_REGIONS


def prioritize_actions(results: dict) -> pd.DataFrame:
    """
    Flatten all actions from all analyses into a single prioritized table.

    Returns DataFrame: [source, branch, insight, action, priority, category]
    """
    all_actions = []

    source_map = {
        "cross_sell": "Cross-Sell",
        "modifier_upsell": "Modifier Upsell",
        "bcg": "Menu Engineering",
        "bundles": "Bundle/Combo",
        "playbooks": "Branch Playbook",
        "seasonal": "Seasonal Strategy",
        "channels": "Channel Expansion",
        "size_upsell": "Size Upsell",
    }

    for key, label in source_map.items():
        if key not in results:
            continue
        actions = results[key].get("actions", [])
        for a in actions:
            row = {
                "source": label,
                "branch": a.get("branch", a.get("season", "Chain-wide")),
                "insight": a.get("insight", ""),
                "action": a.get("action", ""),
                "priority": a.get("priority", "MEDIUM"),
                "category": a.get("category", label),
            }
            all_actions.append(row)

    df = pd.DataFrame(all_actions)

    # Sort: HIGH first, then MEDIUM
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_sort"] = df["priority"].map(priority_order)
    df = df.sort_values(["_sort", "source"]).drop(columns="_sort").reset_index(drop=True)

    return df


def branch_action_cards(results: dict) -> dict:
    """
    Generate per-branch action cards combining all analyses.

    Returns dict of branch -> {metrics, actions, priority_score}
    """
    actions_df = prioritize_actions(results)
    playbooks = results.get("playbooks", {}).get("playbooks", {})

    cards = {}
    for branch, pb in playbooks.items():
        branch_actions = actions_df[actions_df["branch"] == branch]
        high_count = len(branch_actions[branch_actions["priority"] == "HIGH"])

        cards[branch] = {
            "region": pb.get("region", "Unknown"),
            "revenue": pb.get("total_revenue", 0),
            "profit_margin": pb.get("profit_margin", 0),
            "food_attach_rate": pb.get("food_attach_rate", 0),
            "mod_attach_rate": pb.get("mod_attach_rate", 0),
            "toters_pct": pb.get("toters_pct", 0),
            "peak_month": pb.get("peak_month", "N/A"),
            "high_priority_actions": high_count,
            "total_actions": len(branch_actions),
            "actions": branch_actions[["source", "insight", "action", "priority"]].to_dict(
                "records"
            ),
        }

    return cards


def executive_summary(results: dict) -> str:
    """
    Generate a text executive summary of top findings and actions.
    """
    actions_df = prioritize_actions(results)
    high_actions = actions_df[actions_df["priority"] == "HIGH"]

    lines = [
        "=" * 70,
        "STORIES COFFEE — REVENUE ACTION ENGINE: EXECUTIVE SUMMARY",
        "=" * 70,
        "",
        f"Total actionable recommendations: {len(actions_df)}",
        f"HIGH priority: {len(high_actions)}",
        f"MEDIUM priority: {len(actions_df) - len(high_actions)}",
        "",
        "--- TOP HIGH-PRIORITY ACTIONS ---",
        "",
    ]

    # Group by source
    for source in high_actions["source"].unique():
        src_actions = high_actions[high_actions["source"] == source]
        lines.append(f"[{source}] ({len(src_actions)} actions)")
        for _, row in src_actions.head(3).iterrows():
            lines.append(f"  • {row['branch']}: {row['insight']}")
            lines.append(f"    → {row['action']}")
        lines.append("")

    # Key metrics
    if "cross_sell" in results:
        cs = results["cross_sell"]
        lines.append(f"Chain avg food attachment rate: {cs['avg_attach_rate']:.0f}%")
    if "modifier_upsell" in results:
        mu = results["modifier_upsell"]
        lines.append(f"Chain avg modifier attach rate: {mu['avg_attach_rate']:.0f}%")
    if "size_upsell" in results:
        su = results["size_upsell"]
        lines.append(
            f"Size upsell potential (10% upgrade): {su['upsell_10pct_revenue']:,.0f}"
        )
    if "channels" in results:
        ch = results["channels"]
        lines.append(
            f"Branches with <1% Toters: {len(ch['no_toters'])}"
        )

    lines.extend(["", "=" * 70])
    return "\n".join(lines)
