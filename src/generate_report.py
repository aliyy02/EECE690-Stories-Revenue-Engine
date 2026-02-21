"""
Generate executive summary PDF for Stories Coffee Revenue Action Engine.

Usage: python -m src.generate_report
Output: reports/executive_summary.pdf
"""

from pathlib import Path

from fpdf import FPDF

from src.revenue_analysis import run_all_analyses
from src.ml_models import run_all_models
from src.action_engine import prioritize_actions


OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def _sanitize(text):
    """Replace non-latin1 characters for fpdf compatibility."""
    replacements = {
        "\u2014": "--",  # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "-",   # bullet
        "\u2026": "...", # ellipsis
        "\u00b2": "2",   # superscript 2
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Fallback: replace any remaining non-latin1
    return text.encode("latin-1", errors="replace").decode("latin-1")

# Brand colors (RGB)
C_GREEN = (27, 58, 45)
C_GOLD = (200, 169, 110)
C_CREAM = (245, 240, 235)
C_DARK = (51, 51, 51)


class StorysPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_GREEN)
        self.cell(0, 8, _sanitize("Stories Coffee  |  Revenue Action Engine"), align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*C_GREEN)
        self.cell(0, 10, _sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_GOLD)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_DARK)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(2)

    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_DARK)
        self.cell(6, 5.5, "-")
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.write(5.5, _sanitize(bold_prefix) + " ")
            self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(1)


def generate_executive_summary():
    """Generate a 2-page executive summary PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running analyses for report...")
    analyses = run_all_analyses()
    ml = run_all_models()
    actions_df = prioritize_actions(analyses)

    high_actions = actions_df[actions_df["priority"] == "HIGH"]
    total_actions = len(actions_df)
    n_high = len(high_actions)

    cs = analyses["cross_sell"]
    mu = analyses["modifier_upsell"]
    su = analyses["size_upsell"]
    ch = analyses["channels"]
    fc = ml["forecast"]

    pdf = StorysPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1 ──────────────────────────────────────────────────────────
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*C_GREEN)
    pdf.cell(0, 12, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*C_GOLD)
    pdf.cell(0, 7, "Revenue Action Engine  |  Full Year 2025 Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*C_DARK)
    pdf.cell(0, 6, "Prepared by: Ali Yaakoub, Ali Nasrallah, Malek Bakkar", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Problem Statement
    pdf.section_title("Business Context")
    pdf.body_text(
        "Stories Coffee operates 25 branches across Lebanon. This analysis transforms "
        "a full year of POS sales data (2025) into specific, prioritized actions to "
        "increase revenue. Every recommendation is paired with a concrete implementation step."
    )

    # Key Numbers
    pdf.section_title("At a Glance")
    pdf.bullet(f"Total actionable recommendations generated: {total_actions}")
    pdf.bullet(f"HIGH priority (act this week): {n_high}")
    pdf.bullet(f"Chain average food attachment rate: {cs['avg_attach_rate']:.0f}%")
    pdf.bullet(f"Chain average modifiers per 100 drinks: {mu['avg_attach_rate']:.0f}")
    pdf.bullet(
        f"Branches with no delivery presence: {len(ch['no_toters'])} of 25"
    )
    pdf.bullet(
        f"Forecast model accuracy (in-sample R-squared): {fc['train_r2']:.2f}"
    )
    pdf.ln(2)

    # Top 5 Findings
    pdf.section_title("Top 5 Findings")

    findings = [
        (
            "Food Cross-Sell Gap:",
            f"Food attachment rates vary from under 40% to over 80% across branches. "
            f"Closing the gap to the {cs['avg_attach_rate']:.0f}% chain average through "
            f"staff pairing scripts would add thousands of food items sold per year."
        ),
        (
            "Modifier Upsell Variance:",
            f"Top branches sell {mu['avg_attach_rate']:.0f}+ modifiers per 100 drinks. "
            f"Simple POS prompts ('Would you like an extra shot?') can lift lagging branches."
        ),
        (
            "Menu Bloat:",
            f"The BCG analysis identified {len(analyses['bcg']['kill_list'])} low-volume, "
            f"low-margin products. Removing them simplifies operations and reduces waste."
        ),
        (
            "Delivery White Space:",
            f"{len(ch['no_toters'])} branches have virtually no Toters delivery revenue. "
            f"Where active, delivery averages {ch['avg_toters_pct']:.1f}% of branch revenue."
        ),
        (
            "Size Upsell:",
            f"A 10% shift from Small to Medium/Large would generate approximately "
            f"{su['upsell_10pct_revenue']:,.0f} in additional revenue at near-zero incremental cost."
        ),
    ]

    for bold, text in findings:
        pdf.bullet(text, bold_prefix=bold)

    # Methodology
    pdf.section_title("Methodology")
    pdf.body_text(
        "8 revenue analyses (cross-sell, modifiers, BCG menu matrix, bundles, "
        "branch playbooks, seasonal strategy, channel expansion, size upsell) plus "
        "3 ML models (Ridge Regression forecast, K-Means branch clustering, "
        "Menu Engineering classification with composite scoring)."
    )

    # ── Page 2 ──────────────────────────────────────────────────────────
    pdf.add_page()

    pdf.section_title("Top Recommendations by Category")

    # Group HIGH actions by source, show top 2 per source
    for source in high_actions["source"].unique()[:6]:
        src_df = high_actions[high_actions["source"] == source]
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*C_GREEN)
        pdf.cell(0, 7, _sanitize(f"{source} ({len(src_df)} urgent actions)"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*C_DARK)
        for _, row in src_df.head(2).iterrows():
            branch = row.get("branch", "Chain-wide")
            pdf.bullet(
                f"{row['insight']}  ->  {row['action']}",
                bold_prefix=f"[{branch}]"
            )
        pdf.ln(2)

    pdf.section_title("Expected Impact")
    pdf.body_text(
        "These recommendations target the highest-leverage revenue drivers: "
        "cross-sell conversion, modifier upselling, menu rationalization, "
        "delivery channel expansion, and size optimization. "
        "Implementation requires minimal capital investment — primarily "
        "staff training, POS configuration, and menu board updates."
    )

    pdf.bullet(
        "Cross-sell and modifier training can begin immediately at all branches.",
        bold_prefix="Quick Wins:"
    )
    pdf.bullet(
        "Menu kill list review and Toters onboarding require management approval.",
        bold_prefix="Medium-Term:"
    )
    pdf.bullet(
        "Seasonal staffing plans and bundle pricing should be set quarterly.",
        bold_prefix="Strategic:"
    )

    pdf.ln(4)

    pdf.section_title("Forecast Caveat")
    pdf.body_text(
        "The sales forecast model was trained on 1 year of data with in-sample evaluation. "
        "R-squared reflects training fit, not true predictive power. "
        "Use forecasts for directional staffing and inventory planning only."
    )

    # Save
    output_path = OUTPUT_DIR / "executive_summary.pdf"
    pdf.output(str(output_path))
    print(f"\nExecutive summary saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_executive_summary()
