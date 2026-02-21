"""
Parsers for Stories Coffee POS data exports.

Each parser handles a specific CSV file format, dealing with:
- UTF-8 BOM encoding
- Repeated page headers
- Hierarchical data structures
- Comma-formatted numbers
- Inconsistent branch names
"""

import csv
import io
import re
from pathlib import Path

import pandas as pd

from src.utils import (
    BRANCH_NAME_MAP,
    CATEGORIES,
    KNOWN_DIVISIONS,
    SERVICE_TYPES,
    is_branch_name,
    is_column_header,
    is_copyright_line,
    is_page_header,
    normalize_branch_name,
    parse_numeric,
)


def _read_lines(filepath: str | Path) -> list[str]:
    """Read file lines with BOM handling."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return f.readlines()


# =============================================================================
# Parser 1: Monthly Sales (REP_S_00134_SMRY.csv)
# =============================================================================

def parse_monthly_sales(filepath: str | Path) -> pd.DataFrame:
    """
    Parse monthly sales by branch (File 1).

    The file has a weird split-block layout:
    - Lines 4-31: Jan-Sep data for 2025 branches, then 2026 branches
    - Lines 58+: Oct-Dec + Total data for same branches
    - Year appears only on first row of each block, rest are empty -> forward-fill
    - Column headers repeat at line 17, 32, 47, etc.

    Returns DataFrame: [branch, year, month, sales]
    """
    lines = _read_lines(filepath)

    # Parse into two blocks: jan_sep and oct_dec_total
    jan_sep_rows = []
    oct_dec_rows = []
    current_block = None
    current_year = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip title/report header lines
        if line.startswith('Stories,') and 'Month' not in line:
            continue
        if line.startswith('Comparative'):
            continue
        if is_page_header(line):
            continue
        if is_copyright_line(line):
            continue

        # Parse as CSV
        reader = csv.reader(io.StringIO(line))
        fields = next(reader)

        # Skip column header rows
        if len(fields) > 2 and fields[0] == '' and ('January' in fields or 'October' in fields):
            # Detect block type
            if 'January' in fields:
                current_block = 'jan_sep'
            elif 'October' in fields:
                current_block = 'oct_dec'
            continue

        # Skip if fields[1] starts with "January" (another header variant)
        if len(fields) > 1 and fields[1].strip() == '' and len(fields) > 2:
            joined = ','.join(fields)
            if 'January' in joined and 'February' in joined:
                current_block = 'jan_sep'
                continue
            if 'October' in joined and 'November' in joined:
                current_block = 'oct_dec'
                continue

        # Detect year
        if fields[0].strip() in ('2025', '2026'):
            current_year = int(fields[0].strip())

        # Get branch name (field[1] for jan_sep first block, field[0] or field[1] for others)
        branch_raw = fields[1].strip() if len(fields) > 1 else ''
        if not branch_raw and fields[0].strip() not in ('2025', '2026', '', 'Total'):
            branch_raw = fields[0].strip()

        # Skip Total rows and empty rows
        if branch_raw == 'Total' or branch_raw == '':
            if fields[0].strip() == '' and len(fields) > 1 and fields[1].strip() == 'Total':
                continue
            if branch_raw == 'Total':
                continue
            if fields[0].strip() in ('2025', '2026') and len(fields) > 1 and fields[1].strip() == '':
                # Year row with no branch - skip
                continue
            continue

        if not is_branch_name(branch_raw):
            continue

        branch = normalize_branch_name(branch_raw)

        if current_block == 'jan_sep':
            # Fields after branch: may start at index 2 or 3 depending on block
            # Find the numeric values (there should be 9: Jan through Sep)
            nums = []
            for f in fields[2:]:
                f = f.strip()
                if f == '':
                    continue
                nums.append(parse_numeric(f))

            if len(nums) >= 9:
                jan_sep_rows.append({
                    'branch': branch,
                    'year': current_year,
                    'January': nums[0], 'February': nums[1], 'March': nums[2],
                    'April': nums[3], 'May': nums[4], 'June': nums[5],
                    'July': nums[6], 'August': nums[7], 'September': nums[8],
                })

        elif current_block == 'oct_dec':
            nums = []
            for f in fields[2:]:
                f = f.strip()
                if f == '':
                    continue
                nums.append(parse_numeric(f))

            if len(nums) >= 4:
                oct_dec_rows.append({
                    'branch': branch,
                    'year': current_year,
                    'October': nums[0], 'November': nums[1], 'December': nums[2],
                    'total_by_year': nums[3],
                })
            elif len(nums) >= 3:
                oct_dec_rows.append({
                    'branch': branch,
                    'year': current_year,
                    'October': nums[0], 'November': nums[1], 'December': nums[2],
                    'total_by_year': 0.0,
                })

    # Build DataFrames and merge
    df_js = pd.DataFrame(jan_sep_rows)
    df_od = pd.DataFrame(oct_dec_rows)

    if df_js.empty or df_od.empty:
        raise ValueError("Failed to parse monthly sales - one or both blocks empty")

    df = pd.merge(df_js, df_od, on=['branch', 'year'], how='outer')

    # Melt to long format
    month_cols = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']

    df_long = df.melt(
        id_vars=['branch', 'year', 'total_by_year'],
        value_vars=month_cols,
        var_name='month',
        value_name='sales'
    )

    # Add month number for sorting
    month_num = {m: i+1 for i, m in enumerate(month_cols)}
    df_long['month_num'] = df_long['month'].map(month_num)
    df_long = df_long.sort_values(['branch', 'year', 'month_num']).reset_index(drop=True)

    # Fill NaN sales with 0
    df_long['sales'] = df_long['sales'].fillna(0.0)
    df_long['total_by_year'] = df_long['total_by_year'].fillna(0.0)

    return df_long


# =============================================================================
# Parser 2: Product Profitability (rep_s_00014_SMRY.csv)
# =============================================================================

def parse_product_profitability(filepath: str | Path) -> pd.DataFrame:
    """
    Parse product-level profitability data (File 2).

    Uses a state machine to track hierarchical context:
    Branch > Service Type > Category > Division > Products

    Returns DataFrame with columns:
    [branch, service_type, category, division, product, qty, revenue, cost,
     profit, cost_pct, profit_pct, row_type]
    """
    lines = _read_lines(filepath)
    rows = []

    current_branch = None
    current_service = None
    current_category = None
    current_division = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip headers
        if is_page_header(line):
            continue
        if is_copyright_line(line):
            continue

        # Parse CSV
        reader = csv.reader(io.StringIO(line))
        fields = next(reader)
        fields = [f.strip() for f in fields]

        if not fields or not fields[0]:
            continue

        first = fields[0]

        # Skip column headers
        if first == 'Product Desc':
            continue
        if first == 'Stories' and len(fields) > 1 and fields[1] == '':
            # Title row "Stories,,,,..."
            continue

        # Check for branch name
        if is_branch_name(first):
            current_branch = normalize_branch_name(first)
            current_service = None
            current_category = None
            current_division = None
            continue

        # Check for service type
        if first in SERVICE_TYPES:
            current_service = first
            current_category = None
            current_division = None
            continue

        # Check for category
        if first in CATEGORIES:
            current_category = first
            current_division = None
            continue

        # Check for division
        if first in KNOWN_DIVISIONS:
            current_division = first
            continue

        # Check for subtotal rows
        if first.startswith('Total By ') or first.startswith('Total by '):
            row_type = first.rstrip(':').strip()
            # Parse numeric fields
            qty = parse_numeric(fields[1]) if len(fields) > 1 else 0
            total_price = parse_numeric(fields[2]) if len(fields) > 2 else 0
            cost = parse_numeric(fields[4]) if len(fields) > 4 else 0
            cost_pct = parse_numeric(fields[5]) if len(fields) > 5 else 0
            profit = parse_numeric(fields[6]) if len(fields) > 6 else 0
            profit_pct = parse_numeric(fields[8]) if len(fields) > 8 else 0
            # Fix Total Price truncation: use cost + profit for revenue
            revenue = cost + profit

            rows.append({
                'branch': current_branch,
                'service_type': current_service,
                'category': current_category,
                'division': current_division,
                'product': row_type,
                'qty': qty,
                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'cost_pct': cost_pct,
                'profit_pct': profit_pct,
                'row_type': 'subtotal',
            })

            # Reset context based on subtotal level
            if 'Division' in row_type:
                current_division = None
            elif 'Category' in row_type:
                current_category = None
                current_division = None
            elif 'Department' in row_type:
                current_service = None
                current_category = None
                current_division = None
            elif 'Branch' in row_type:
                current_branch = None
                current_service = None
                current_category = None
                current_division = None
            continue

        # Product rows - must have numeric data
        if current_branch and len(fields) > 6:
            qty = parse_numeric(fields[1])
            total_price = parse_numeric(fields[2])
            cost = parse_numeric(fields[4]) if len(fields) > 4 else 0
            cost_pct = parse_numeric(fields[5]) if len(fields) > 5 else 0
            profit = parse_numeric(fields[6]) if len(fields) > 6 else 0
            profit_pct = parse_numeric(fields[8]) if len(fields) > 8 else 0

            # For product-level rows, Total Price is reliable
            revenue = total_price if total_price != 0 else (cost + profit)

            if qty != 0 or cost != 0 or profit != 0:
                rows.append({
                    'branch': current_branch,
                    'service_type': current_service,
                    'category': current_category,
                    'division': current_division,
                    'product': first,
                    'qty': qty,
                    'revenue': revenue,
                    'cost': cost,
                    'profit': profit,
                    'cost_pct': cost_pct,
                    'profit_pct': profit_pct,
                    'row_type': 'product',
                })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# Parser 3: Sales by Group (rep_s_00191_SMRY-3.csv)
# =============================================================================

def parse_sales_by_group(filepath: str | Path) -> pd.DataFrame:
    """
    Parse sales by product group (File 3).

    Structure uses explicit prefixes: "Branch: ...", "Division: ...", "Group: ..."

    Returns DataFrame: [branch, division, group, product, qty, revenue, row_type]
    """
    lines = _read_lines(filepath)
    rows = []

    current_branch = None
    current_division = None
    current_group = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_page_header(line):
            continue
        if is_copyright_line(line):
            continue

        reader = csv.reader(io.StringIO(line))
        fields = next(reader)
        fields = [f.strip() for f in fields]

        if not fields or not fields[0]:
            continue

        first = fields[0]

        # Skip column headers
        if first == 'Description':
            continue
        if first == 'Stories' and len(fields) > 1 and fields[1] == '':
            continue
        if first == 'Sales by Items By Group':
            continue

        # Context lines
        if first.startswith('Branch: '):
            branch_raw = first[len('Branch: '):]
            current_branch = normalize_branch_name(branch_raw)
            current_division = None
            current_group = None
            continue

        if first.startswith('Division: '):
            current_division = first[len('Division: '):]
            current_group = None
            continue

        if first.startswith('Group: '):
            current_group = first[len('Group: '):]
            continue

        # Subtotal rows
        if first.startswith('Total by Group:'):
            label = first.split(':')[1].strip() if ':' in first else current_group
            qty = parse_numeric(fields[2]) if len(fields) > 2 else 0
            revenue = parse_numeric(fields[3]) if len(fields) > 3 else 0
            rows.append({
                'branch': current_branch,
                'division': current_division,
                'group': current_group or label,
                'product': f'Total by Group: {current_group or label}',
                'qty': qty,
                'revenue': revenue,
                'row_type': 'group_total',
            })
            continue

        if first.startswith('Total by Division:'):
            qty = parse_numeric(fields[2]) if len(fields) > 2 else 0
            revenue = parse_numeric(fields[3]) if len(fields) > 3 else 0
            rows.append({
                'branch': current_branch,
                'division': current_division,
                'group': None,
                'product': f'Total by Division: {current_division}',
                'qty': qty,
                'revenue': revenue,
                'row_type': 'division_total',
            })
            current_division = None
            current_group = None
            continue

        if first.startswith('Total by Branch:'):
            qty = parse_numeric(fields[2]) if len(fields) > 2 else 0
            revenue = parse_numeric(fields[3]) if len(fields) > 3 else 0
            rows.append({
                'branch': current_branch,
                'division': None,
                'group': None,
                'product': f'Total by Branch: {current_branch}',
                'qty': qty,
                'revenue': revenue,
                'row_type': 'branch_total',
            })
            current_branch = None
            current_division = None
            current_group = None
            continue

        # Product rows
        if current_branch and len(fields) >= 4:
            qty = parse_numeric(fields[2])
            revenue = parse_numeric(fields[3])
            if qty != 0 or revenue != 0:
                rows.append({
                    'branch': current_branch,
                    'division': current_division,
                    'group': current_group,
                    'product': first,
                    'qty': qty,
                    'revenue': revenue,
                    'row_type': 'product',
                })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# Parser 4: Category Summary (rep_s_00673_SMRY.csv)
# =============================================================================

def parse_category_summary(filepath: str | Path) -> pd.DataFrame:
    """
    Parse category profit summary by branch (File 4).

    Simple structure: Branch name on its own line, then BEVERAGES and FOOD rows,
    then Total By Branch row.

    Returns DataFrame: [branch, category, qty, revenue, cost, profit, cost_pct, profit_pct]
    """
    lines = _read_lines(filepath)
    rows = []

    current_branch = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_page_header(line):
            continue
        if is_copyright_line(line):
            continue

        reader = csv.reader(io.StringIO(line))
        fields = next(reader)
        fields = [f.strip() for f in fields]

        if not fields or not fields[0]:
            continue

        first = fields[0]

        # Skip headers
        if first == 'Category':
            continue
        if first == 'Stories' and len(fields) > 1 and fields[1] == '':
            continue
        if first == 'Theoretical Profit By Category':
            continue

        # Branch name line (alone, all other fields empty)
        if is_branch_name(first):
            current_branch = normalize_branch_name(first)
            continue

        # Category rows (BEVERAGES or FOOD)
        if first in ('BEVERAGES', 'FOOD') and current_branch:
            qty = parse_numeric(fields[1]) if len(fields) > 1 else 0
            # Total Price is here but may be truncated - use cost + profit
            cost = parse_numeric(fields[4]) if len(fields) > 4 else 0
            cost_pct = parse_numeric(fields[5]) if len(fields) > 5 else 0
            profit = parse_numeric(fields[6]) if len(fields) > 6 else 0
            profit_pct = parse_numeric(fields[8]) if len(fields) > 8 else 0
            revenue = cost + profit

            rows.append({
                'branch': current_branch,
                'category': first,
                'qty': qty,
                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'cost_pct': cost_pct,
                'profit_pct': profit_pct,
                'row_type': 'category',
            })
            continue

        # Total By Branch row
        if first.startswith('Total By Branch:'):
            qty = parse_numeric(fields[1]) if len(fields) > 1 else 0
            cost = parse_numeric(fields[4]) if len(fields) > 4 else 0
            cost_pct = parse_numeric(fields[5]) if len(fields) > 5 else 0
            profit = parse_numeric(fields[6]) if len(fields) > 6 else 0
            profit_pct = parse_numeric(fields[8]) if len(fields) > 8 else 0
            revenue = cost + profit

            rows.append({
                'branch': current_branch,
                'category': 'TOTAL',
                'qty': qty,
                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'cost_pct': cost_pct,
                'profit_pct': profit_pct,
                'row_type': 'branch_total',
            })
            current_branch = None
            continue

    df = pd.DataFrame(rows)
    return df
