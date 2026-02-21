# Stories Coffee — Revenue Action Engine

A data-driven decision engine for **Stories Coffee**, one of Lebanon's fastest-growing coffee chains with 25 branches. Built for the Introduction to Machine Learning 12-Hour Hackathon.

## Business Problem

> *"I have all this data... but I don't know what to do with it. Tell me how to make more money."*

We transformed a year of raw POS data into **actionable, branch-specific revenue recommendations** — not just charts, but concrete actions with priorities.

## Key Findings

| Finding | Impact |
|---------|--------|
| **Food cross-sell gap** | Many branches sell far fewer food items per drink than top performers. Staff training on pairing scripts can close the gap. |
| **Modifier upsell opportunity** | Some branches average 2+ paid add-ons per drink while others lag behind. Simple POS prompts drive this. |
| **Menu bloat** | Dozens of low-volume, low-margin items waste prep time and confuse customers. A kill list of bottom performers is provided. |
| **Delivery white space** | Most branches have zero Toters (delivery) presence despite proven demand at active locations. |
| **Seasonal planning** | May-June revenue drops sharply. Pre-planned cost cuts and promotions can protect margins. |
| **Size upsell** | A 10% shift from Small to Medium/Large drinks yields significant incremental revenue at near-zero cost. |

## Approach & Methodology

### 8 Revenue Analyses
1. **Cross-Sell Opportunity Matrix** — Food attachment rate by branch
2. **Modifier Upsell Engine** — Add-on rates and top modifiers
3. **Menu Kill List (BCG Matrix)** — Star/Dog/Puzzle/Plow Horse classification
4. **Bundle/Combo Builder** — Optimal food+drink pairings with margin analysis
5. **Branch-Specific Playbooks** — Per-branch scorecards vs chain averages
6. **Seasonal Revenue Strategy** — Monthly patterns and trough resilience
7. **Channel Expansion** — Toters delivery penetration analysis
8. **Size Upsell Opportunity** — Size distribution and upgrade potential

### 3 ML Models
1. **Sales Forecasting** — Ridge Regression with seasonal features (Jan-Jun 2026)
2. **Branch Clustering** — K-Means segmentation into operational groups
3. **Menu Engineering Classification** — BCG quadrant with composite Menu Power score

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Overview** | KPIs, urgent action count, action distribution |
| **Revenue Levers** | Cross-sell, modifiers, size upsell, delivery channels |
| **Menu Engineering** | Product classification scatter, kill list, combo builder |
| **Branch Playbooks** | Per-branch scorecards, comparison charts, cluster analysis |
| **Forecasting & Seasonal** | Sales forecast with disclaimers, seasonal patterns |
| **Upload New Data** | CSV upload UI for refreshing the dashboard with new exports |

## How to Run

### Quick Start
```bash
pip install -r requirements.txt
python -m src.cleaning          # Process raw CSVs into parquet
streamlit run app/streamlit_app.py
```

### With Docker
```bash
docker build -t stories-coffee .
docker run -p 8501:8501 stories-coffee
```

Then open http://localhost:8501

## Project Structure

```
ML/
├── app/
│   └── streamlit_app.py        # 6-page Streamlit dashboard
├── src/
│   ├── data_loader.py          # CSV parsers for all 4 POS report formats
│   ├── cleaning.py             # Data pipeline orchestrator
│   ├── revenue_analysis.py     # 8 revenue analysis functions
│   ├── ml_models.py            # 3 ML models (forecast, clustering, BCG)
│   ├── action_engine.py        # Action prioritization and branch cards
│   ├── utils.py                # Branch name mapping, region mapping
│   └── generate_report.py      # Executive summary PDF generator
├── data/processed/             # Cleaned parquet files (gitignored)
├── Stories_data/               # Raw CSV exports (gitignored)
├── knowledge_base/             # Reference materials
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11** — pandas, numpy, scikit-learn
- **Plotly** — Interactive charts with brand colors
- **Streamlit** — Dashboard framework
- **fpdf2** — PDF report generation

## Data

Raw CSV exports and processed parquet files are included in the repository for reproducibility:

| File | Description | Rows |
|------|-------------|------|
| `Stories_data/REP_S_00134_SMRY.csv` | Monthly sales by branch (YoY) | 111 |
| `Stories_data/rep_s_00014_SMRY.csv` | Product-level profitability | 14,584 |
| `Stories_data/rep_s_00191_SMRY-3.csv` | Sales by product groups | 14,139 |
| `Stories_data/rep_s_00673_SMRY.csv` | Category profit summary | 108 |

Pre-processed parquets are in `data/processed/` so the dashboard works immediately without running the cleaning pipeline. To re-process from raw CSVs, run `python -m src.cleaning`. New data can also be uploaded through the dashboard's **Upload New Data** page.
