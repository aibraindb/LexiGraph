import re, datetime as dt
from typing import Optional

MONEY_RE = re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+\.[0-9]{2})")
DATE_RE = re.compile(r"([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")

def parse_money(s: str) -> Optional[str]:
    m = MONEY_RE.search(s or "")
    return m.group(1) if m else None

def parse_date(s: str) -> Optional[str]:
    m = DATE_RE.search(s or "")
    if not m: return None
    mm, dd, yy = m.group(1).split('/')
    yy = int(yy); yy = yy + 2000 if yy < 100 else yy
    try:
        return dt.date(int(yy), int(mm), int(dd)).isoformat()
    except Exception:
        return None

def mask_routing(s: str) -> str:
    return re.sub(r"\b(\d{9})\b", lambda m: "******"+m.group(1)[-3:], s or "")
