"""
Data cleaning orchestrator.

Runs all 4 parsers, validates, and exports to parquet.
Usage: python -m src.cleaning
"""

from pathlib import Path

from src.data_loader import (
    parse_category_summary,
    parse_monthly_sales,
    parse_product_profitability,
    parse_sales_by_group,
)

DATA_DIR = Path(__file__).parent.parent / 'Stories_data'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'processed'


def run_cleaning(data_dir=None, output_dir=None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STORIES COFFEE DATA CLEANING PIPELINE")
    print("=" * 60)

    # --- File 1: Monthly Sales ---
    print("\n[1/4] Parsing monthly sales...")
    df_monthly = parse_monthly_sales(data_dir / 'REP_S_00134_SMRY.csv')
    print(f"  Rows: {len(df_monthly)}")
    print(f"  Branches: {df_monthly['branch'].nunique()}")
    print(f"  Years: {sorted(df_monthly['year'].unique())}")
    print(f"  2025 total: {df_monthly[df_monthly['year']==2025]['sales'].sum():,.0f}")
    df_monthly.to_parquet(output_dir / 'monthly_sales.parquet', index=False)
    print("  -> Saved monthly_sales.parquet")

    # --- File 2: Product Profitability ---
    print("\n[2/4] Parsing product profitability...")
    df_products = parse_product_profitability(data_dir / 'rep_s_00014_SMRY.csv')
    products_only = df_products[df_products['row_type'] == 'product']
    print(f"  Total rows: {len(df_products)}")
    print(f"  Product rows: {len(products_only)}")
    print(f"  Subtotal rows: {len(df_products) - len(products_only)}")
    print(f"  Branches: {df_products['branch'].nunique()}")
    print(f"  Unique products: {products_only['product'].nunique()}")
    print(f"  Service types: {sorted(df_products['service_type'].dropna().unique())}")
    df_products.to_parquet(output_dir / 'product_profitability.parquet', index=False)
    print("  -> Saved product_profitability.parquet")

    # --- File 3: Sales by Group ---
    print("\n[3/4] Parsing sales by group...")
    df_groups = parse_sales_by_group(data_dir / 'rep_s_00191_SMRY-3.csv')
    groups_only = df_groups[df_groups['row_type'] == 'product']
    print(f"  Total rows: {len(df_groups)}")
    print(f"  Product rows: {len(groups_only)}")
    print(f"  Branches: {df_groups['branch'].nunique()}")
    print(f"  Groups: {df_groups['group'].nunique()}")
    df_groups.to_parquet(output_dir / 'sales_by_group.parquet', index=False)
    print("  -> Saved sales_by_group.parquet")

    # --- File 4: Category Summary ---
    print("\n[4/4] Parsing category summary...")
    df_category = parse_category_summary(data_dir / 'rep_s_00673_SMRY.csv')
    cat_only = df_category[df_category['row_type'] == 'category']
    print(f"  Total rows: {len(df_category)}")
    print(f"  Branches: {df_category['branch'].nunique()}")
    bev = cat_only[cat_only['category'] == 'BEVERAGES']
    food = cat_only[cat_only['category'] == 'FOOD']
    print(f"  Avg beverage margin: {bev['profit_pct'].mean():.1f}%")
    print(f"  Avg food margin: {food['profit_pct'].mean():.1f}%")
    df_category.to_parquet(output_dir / 'category_summary.parquet', index=False)
    print("  -> Saved category_summary.parquet")

    # --- Cross-validation ---
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    branches_f1 = set(df_monthly['branch'].unique())
    branches_f2 = set(df_products['branch'].dropna().unique())
    branches_f4 = set(df_category['branch'].unique())

    print(f"  File 1 branches: {len(branches_f1)}")
    print(f"  File 2 branches: {len(branches_f2)}")
    print(f"  File 4 branches: {len(branches_f4)}")

    all_branches = branches_f1 | branches_f2 | branches_f4
    print(f"  All unique branches: {sorted(all_branches)}")

    # Check File 4 totals vs File 1 totals
    f4_totals = df_category[df_category['row_type'] == 'branch_total']
    f4_revenue = f4_totals['revenue'].sum()
    f1_revenue = df_monthly[df_monthly['year'] == 2025]['total_by_year'].drop_duplicates().sum()

    print(f"\n  File 4 total revenue (cost+profit): {f4_revenue:,.0f}")
    print(f"  File 1 total 2025 revenue: {f1_revenue:,.0f}")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_cleaning()
