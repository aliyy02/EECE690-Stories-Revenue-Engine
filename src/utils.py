"""Utility functions for Stories Coffee data processing."""

import re

# Canonical branch name mapping (raw POS name -> clean short name)
BRANCH_NAME_MAP = {
    "Stories - Bir Hasan": "Bir Hasan",
    "Stories Ain El Mreisseh": "Ain El Mreisseh",
    "Stories Airport": "Airport",
    "Stories Antelias": "Antelias",
    "Stories Batroun": "Batroun",
    "Stories Bayada": "Bayada",
    "Stories Centro Mall": "Centro Mall",
    "Stories Event Starco": "Event Starco",
    "Stories Faqra": "Faqra",
    "Stories Khaldeh": "Khaldeh",
    "Stories LAU": "LAU",
    "Stories Le Mall": "Le Mall",
    "Stories Mansourieh": "Mansourieh",
    "Stories Ramlet El Bayda": "Ramlet El Bayda",
    "Stories Saida": "Saida",
    "Stories Sour 2": "Sour 2",
    "Stories Verdun": "Verdun",
    "Stories Zalka": "Zalka",
    "Stories alay": "Aley",
    "Stories amioun": "Amioun",
    "Stories jbeil": "Jbeil",
    "Stories kaslik": "Kaslik",
    "Stories raouche": "Raouche",
    "Stories sin el fil": "Sin El Fil",
    "Stories.": "Unknown/Closed",
}

BRANCH_REGIONS = {
    "Ain El Mreisseh": "Beirut Central",
    "Verdun": "Beirut Central",
    "Raouche": "Beirut Central",
    "Ramlet El Bayda": "Beirut Central",
    "Bayada": "Beirut Central",
    "Sin El Fil": "Greater Beirut",
    "Bir Hasan": "Greater Beirut",
    "Khaldeh": "Greater Beirut",
    "Mansourieh": "Greater Beirut",
    "Batroun": "North",
    "Amioun": "North",
    "Jbeil": "North",
    "Antelias": "Metn",
    "Zalka": "Metn",
    "Kaslik": "Metn",
    "Saida": "South",
    "Sour 2": "South",
    "Faqra": "Mountains",
    "Aley": "Mountains",
    "Centro Mall": "Malls & Special",
    "Le Mall": "Malls & Special",
    "LAU": "Malls & Special",
    "Airport": "Malls & Special",
    "Event Starco": "Malls & Special",
    "Unknown/Closed": "Unknown",
}

# Known division names in File 2
KNOWN_DIVISIONS = {
    "COLD BAR SECTION", "HOT BAR SECTION", "GRAB&GO BEVERAGES",
    "COFFEE PASTRY", "CROISSANT", "FRENCH PASTRY", "GRAB&GO FOOD",
    "SANDWICHES", "DONUTS", "FRUITS", "PACKAGED DESSERT", "SALADS",
}

# Service types in File 2
SERVICE_TYPES = {"TAKE AWAY", "TABLE", "Toters"}

# Categories in File 2
CATEGORIES = {"BEVERAGES", "FOOD"}

# Page header pattern
PAGE_HEADER_RE = re.compile(r'^\d{2}-\w{3}-\d{2}')
PAGE_HEADER_FULL_RE = re.compile(r'^\d{2}-\w{3}-\d{4}')


def normalize_branch_name(raw_name: str) -> str:
    """Convert raw POS branch name to canonical short name."""
    raw_name = raw_name.strip()
    # Strip "Branch: " prefix (used in File 3)
    if raw_name.startswith("Branch: "):
        raw_name = raw_name[len("Branch: "):]
    raw_name = raw_name.strip()
    return BRANCH_NAME_MAP.get(raw_name, raw_name)


def parse_numeric(value) -> float:
    """Parse a numeric value that may have commas, quotes, or be empty."""
    if value is None:
        return 0.0
    s = str(value).strip().strip('"')
    if s == '' or s == '-':
        return 0.0
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return 0.0


def is_page_header(line: str) -> bool:
    """Check if a line is a POS page header."""
    return bool(PAGE_HEADER_RE.match(line)) or bool(PAGE_HEADER_FULL_RE.match(line))


def is_column_header(fields: list, file_type: str) -> bool:
    """Check if a row is a repeated column header."""
    if not fields:
        return False
    first = fields[0].strip()
    if file_type == 'product_profitability':
        return first == 'Product Desc'
    elif file_type == 'sales_by_group':
        return first == 'Description'
    elif file_type == 'category_summary':
        return first == 'Category'
    return False


def is_branch_name(text: str) -> bool:
    """Check if text is a known branch name."""
    text = text.strip()
    return text in BRANCH_NAME_MAP


def is_copyright_line(line: str) -> bool:
    """Check if line is the copyright footer."""
    return 'Copyright' in line or 'Omega Software' in line
