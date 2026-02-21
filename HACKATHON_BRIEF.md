# Stories Coffee Hackathon

## Introduction to Machine Learning — 12-Hour Data Challenge

---

## Overview

This hackathon is not just an assignment; it's a **real consulting gig**.

You have **12 hours** to transform raw business data into actionable insights that could change how a real Lebanese coffee chain operates. No toy datasets. No Titanic survival predictions. **Real data. Real business. Real impact.**

The goal? Show us what you can build when the clock is ticking, the stakes are real, and the only limit is your creativity.

**This is your proving ground.**

---

## Your Mission

You've been hired as a data science consultant by **Stories**, one of Lebanon's fastest-growing coffee chains with **25 branches** across the country.

The founder walks into the room and drops a year's worth of sales data on your desk:

> *"I have all this data... but I don't know what to do with it. Tell me how to make more money."*

That's it. That's the brief. **Figure it out.**

---

## Choose Your Path

There's no single correct answer. You decide how to attack this problem:

| Approach | Examples |
|----------|----------|
| **Exploratory Analysis** | Uncover hidden patterns, seasonality, anomalies |
| **Predictive Modeling** | Forecast sales, predict branch performance, demand planning |
| **Optimization** | Menu engineering, pricing strategy, branch efficiency |
| **Segmentation** | Customer personas, product clustering, location analysis |
| **Interactive Tools** | Dashboards, recommendation systems, decision-support apps |
| **ML/DL Models** | Regression, classification, time series, clustering |

**Mix and match. Go deep or go wide. The most important thing: solve a meaningful problem.**

You're encouraged to use pretrained models, open-source libraries, AI assistants, and online resources. **You're graded on what you build and discover, not what you copy.**

---

## The Data

You have access to **real operational data** from Stories covering **2025 (full year) and January 2026**. All patterns, relationships, and proportions are authentic.

### Important Note

> **The numbers are in arbitrary units.** Focus on patterns, ratios, and relative comparisons — not absolute values.

This is standard practice in data science. For ML tasks, absolute scale doesn't matter:
- **Forecasting?** Patterns drive predictions
- **Clustering?** Relative distances matter
- **Regression?** Coefficients adjust automatically

### Available Files

Located in the `Stories_data/` folder:

| File | What's Inside | Size |
|------|---------------|------|
| `REP_S_00134_SMRY.csv` | Monthly sales by branch (YoY comparison) | ~110 lines |
| `rep_s_00014_SMRY.csv` | Product-level profitability (every item sold) | ~14,600 lines |
| `rep_s_00191_SMRY-3.csv` | Sales by product groups & categories | ~14,100 lines |
| `rep_s_00673_SMRY.csv` | Category profit summary by branch | ~110 lines |

### Heads Up — Data Format

These CSV files are **raw exports from the POS system**, not cleaned datasets. Expect messiness:

- **Page headers are repeated** throughout each file (e.g., `Page 1 of 396`). You'll need to filter these out when parsing.
- **Branch names are inconsistently capitalized** (e.g., `Stories alay` vs. `Stories Ain El Mreisseh`). Normalize them.
- **Some columns contain empty/spacer fields** — artifacts of the export format.
- **The `Total Price` column in Files 2 & 4** has a display truncation bug from the POS system: for larger aggregate values (branch and category totals), the number is silently divided by 10 or 100. The `Total Cost`, `Total Profit`, and all `%` columns are correct. You can derive true revenue as `Total Cost + Total Profit` where needed. Individual product-level rows are not affected.

Dealing with messy real-world data is part of the challenge.

---

## Data Dictionary

### File 1: Monthly Sales (`REP_S_00134_SMRY.csv`)

| Field | Description |
|-------|-------------|
| Year | 2025 or 2026 |
| Branch Name | Location (e.g., "Stories Zalka") |
| January - December | Monthly sales totals |
| Total By Year | Annual revenue per branch |

**Discover:** Seasonality, growth trends, new branch ramp-up, YoY performance

---

### File 2: Product Profitability (`rep_s_00014_SMRY.csv`)

| Field | Description |
|-------|-------------|
| Product Desc | Full item name |
| Qty | Units sold |
| Total Price | Revenue generated |
| Total Cost | Cost of goods sold |
| Total Cost % | Cost as % of revenue |
| Total Profit | Gross profit |
| Total Profit % | Profit margin |

**Structure:**
```
Branch → Service Type (TAKE AWAY / TABLE)
       → Category (BEVERAGES / FOOD)
       → Section (HOT BAR / COLD BAR / DONUTS)
       → Products & Modifiers
```

**Discover:** Margin analysis, modifier popularity, profitable vs. unprofitable items

---

### File 3: Sales by Groups (`rep_s_00191_SMRY-3.csv`)

| Field | Description |
|-------|-------------|
| Description | Product name |
| Barcode | Product barcode (mostly empty) |
| Qty | Units sold |
| Total Amount | Revenue |

**Groups:** BLACK COFFEE, MIXED HOT BEVERAGE, TEA, BLENDED DRINKS, DONUTS, etc.

**Discover:** Category performance, group trends, division comparisons

---

### File 4: Category Summary (`rep_s_00673_SMRY.csv`)

| Field | Description |
|-------|-------------|
| Category | BEVERAGES or FOOD |
| Qty, Total Price, Total Cost, Total Profit | Aggregated metrics |
| Cost %, Profit % | Margin percentages |

**Discover:** Beverages vs Food performance, branch efficiency, category mix

---

## The Playing Field

### 25 Branches Across Lebanon

| Region | Locations |
|--------|-----------|
| **Beirut Central** | Ain El Mreisseh, Verdun, Raouche, Ramlet El Bayda, Bayada |
| **Greater Beirut** | Sin El Fil, Bir Hasan, Khaldeh, Mansourieh |
| **North** | Batroun, Amioun, Jbeil |
| **Metn** | Antelias, Zalka, Kaslik |
| **South** | Saida, Sour 2 |
| **Mountains** | Faqra, Aley |
| **Malls & Special** | Centro Mall, Le Mall, LAU, Airport, Event Starco |

> **Note:** Branch names in the data may differ slightly from this table (e.g., `Stories alay` instead of `Stories Aley`). One branch appears as `Stories.` — likely a temporary or closed location.

### 300+ Products

**Beverages:** Espresso, Latte, Cappuccino, Flat White, Mocha, Caramel Macchiato, Matcha Latte, Frappuccinos, Cold Brew, Double Shot Shaken...

**Food:** Donuts (Pistachio, Boston Cream, Lotus), Croissants, Sandwiches, Pastries, Grab & Go...

**Modifiers:** Extra shots, alternative milks (Oat, Almond, Coconut), sugar-free syrups, whipped cream...

---

## Minimum Requirements

Your submission **must include**:

### 1. Executive Summary (2 pages max, PDF)

A document the CEO could read in 5 minutes:
- **Problem Statement** — What business question did you tackle?
- **Key Findings** — Top 3-5 data-backed insights
- **Recommendations** — Concrete, actionable next steps
- **Expected Impact** — How would this affect the business?
- **Methodology** — Brief description of your approach

### 2. Public GitHub Repository

With a clean, professional `README.md` explaining:
- The business problem
- Your approach and methodology
- How to run/reproduce your analysis
- Key findings and visualizations

**Plus:**
- `requirements.txt` or equivalent (pin your versions)
- Organized, commented code
- Any notebooks, scripts, or dashboards

---

## Bonus Points (The "Wow" Factor)

Go beyond analysis. Build something:

| Bonus | Description |
|-------|-------------|
| **Working Tool** | Dashboard, app, or API that Stories could actually use |
| **Online Deployment** | Publicly accessible (Streamlit, HuggingFace Spaces, etc.) |
| **Future-Proof Design** | Tool works with new data exports without code changes |
| **Docker Container** | Reproducible, deployable solution |
| **Creative Approach** | Unexpected insights, novel techniques |

### If You Build a Tool

It **must** be reproducible:
- Accept CSV files in the provided format
- Work without hardcoded paths or manual data transformation
- Include clear documentation
- A non-technical user should be able to upload next month's data and get results

---

## Team Format

- **Maximum 3 students per team** — one shared grade
- **Solo/Duo submissions allowed** — but same expectations as a team of 3

---

## Grading

This is a **contest-style evaluation**. No rubric — grades are relative to the class.

| Rank | Grade |
|------|-------|
| 1st place | 20/20 |
| 2nd place | 19.5/20 |
| 3rd place | 19/20 |
| ... | (step of 0.5) |

### What We're Looking For

| Criterion | What Matters |
|-----------|--------------|
| **Insight Quality** | Did you find something non-obvious and meaningful? |
| **Business Impact** | Would this actually help Stories make money? |
| **Technical Execution** | Is it sound, reproducible, well-documented? |
| **Communication** | Can a non-technical person understand your findings? |
| **Wow Factor** | Did you surprise us? |

---

## Rules

1. **Use any tools** — Python, R, SQL, Excel, Tableau, whatever works
2. **Use any resources** — Libraries, pretrained models, AI assistants, Stack Overflow
3. **Collaboration encouraged** — Within your team
4. **Questions?** — Figure it out. That's part of the challenge.

---

## Submission

- Submit your **public GitHub repo link** via Moodle
- Submission opens at hackathon start
- Results released within 1 week

---

## Pro Tips

- **Start with EDA** — Understand before you model
- **Think like the owner** — What would make you act?
- **Simple beats complex** — A clear insight > a confusing model
- **Time-box ruthlessly** — 12 hours goes fast
- **Ship something** — A working prototype beats a perfect plan

---

## Final Words

You have 12 hours, real data, and a real business problem.

Don't just analyze, **build**. Don't just describe, **recommend**. Don't just present, **convince**.

Show us what happens when you treat this like a real consulting engagement, not a homework assignment.

**The clock starts now. Make it count.**

---

*This hackathon uses real data from Stories Coffee, provided with permission for educational purposes.*
