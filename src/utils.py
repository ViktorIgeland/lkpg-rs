import re
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup


SWEDISH_MONTHS = {
    "januari": 1,
    "februari": 2,
    "mars": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "augusti": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(html_or_text: str) -> str:
    """Remove HTML and normalize whitespace."""
    soup = BeautifulSoup(html_or_text, "html.parser")
    text = soup.get_text(" ")
    return normalize_whitespace(text)


def try_parse_iso(date_text: str) -> Optional[str]:
    """Try to parse ISO-like date strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM)."""
    date_text = date_text.strip()
    # Direct ISO date
    m = re.search(r"(\d{4}-\d{2}-\d{2})", date_text)
    if m:
        try:
            dt = datetime.fromisoformat(m.group(1))
            return dt.date().isoformat()
        except Exception:
            pass
    # Datetime attribute like 2024-09-01T12:34
    m = re.search(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}", date_text)
    if m:
        try:
            dt = datetime.fromisoformat(m.group(1))
            return dt.date().isoformat()
        except Exception:
            pass
    return None


def try_parse_swedish(date_text: str) -> Optional[str]:
    """Parse dates like '28 augusti 2024'."""
    lower = date_text.lower()
    m = re.search(r"(\d{1,2})\s+([a-zåäö]+)\s+(\d{4})", lower)
    if not m:
        return None
    day_str, month_name, year_str = m.groups()
    month = SWEDISH_MONTHS.get(month_name)
    if not month:
        return None
    try:
        dt = datetime(int(year_str), int(month), int(day_str))
        return dt.date().isoformat()
    except Exception:
        return None


def normalize_date(date_text: str) -> Optional[str]:
    """Normalize various date formats to ISO YYYY-MM-DD, if possible."""
    if not date_text:
        return None
    for fn in (try_parse_iso, try_parse_swedish):
        iso = fn(date_text)
        if iso:
            return iso
    return None


