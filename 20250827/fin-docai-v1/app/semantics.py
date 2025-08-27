
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

ORG_SUFFIXES = [
    "inc","inc.","llc","l.l.c","corp","corp.","co","co.","company","ltd","ltd.","plc","gmbh",
    "s.a.","s.a.s.","s.p.a.","ag","nv","bank","trust","advisors","partners","lp","llp","lllp"
]

STREET_SUFFIX = r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|pkwy|parkway|hwy|highway)"
STATE_2 = r"(?:AL|AK|AS|AZ|AR|CA|CO|CT|DC|DE|FL|GA|GU|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MP|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|PR|RI|SC|SD|TN|TX|UM|UT|VA|VI|VT|WA|WI|WV|WY)"
ZIP_RE = r"\d{5}(?:-\d{4})?"

RE_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
RE_IPV4 = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
RE_IPV6 = re.compile(r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$")
RE_MONEY = re.compile(r"^\s*(?:USD|\$)?\s*[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})\s*$")
RE_DATE_ANY = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")
RE_ADDRESS = re.compile(
    rf"^\s*\d+\s+.+?\b{STREET_SUFFIX}\b(?:[.,]|\s)\s*.+?,\s*{STATE_2}\s+{ZIP_RE}\s*$",
    re.IGNORECASE
)

def norm_money(s: str) -> Optional[float]:
    if not s: return None
    try:
        t = s.replace("$","").replace("USD","").replace(",","").strip()
        return float(t)
    except Exception:
        return None

def is_email(s: str) -> bool:
    return bool(RE_EMAIL.match(s.strip()))

def is_ip(s: str) -> bool:
    s = s.strip()
    return bool(RE_IPV4.match(s) or RE_IPV6.match(s))

def is_money(s: str) -> bool:
    return bool(RE_MONEY.match(s.strip()))

def is_date(s: str) -> Tuple[bool, Optional[str]]:
    t = s.strip()
    if not t: return False, None
    try:
        dt = pd.to_datetime(t, errors="raise", dayfirst=False, infer_datetime_format=True)
        return True, dt.strftime("%Y-%m-%d")
    except Exception:
        # last resort: detect obvious date tokens
        return (bool(RE_DATE_ANY.search(t)), None)

def is_us_address(s: str) -> bool:
    return bool(RE_ADDRESS.match(s.strip()))

def looks_like_person(s: str) -> bool:
    # Basic: 2-4 tokens, capitalized, no digits, not ending with org suffix
    toks = [t for t in re.split(r"\s+", s.strip()) if t]
    if not (2 <= len(toks) <= 4): return False
    if any(any(ch.isdigit() for ch in t) for t in toks): return False
    if any(t.lower().rstrip(".,") in ORG_SUFFIXES for t in toks[-2:]): return False
    # at least two tokens start with uppercase
    cap = sum(1 for t in toks if t and t[0].isupper())
    return cap >= 2

def looks_like_org(s: str) -> bool:
    t = s.strip().lower()
    return any(t.endswith(suf) or f" {suf} " in f" {t} " for suf in ORG_SUFFIXES)

def product_or_service(text: str) -> Optional[str]:
    t = text.lower()
    svc_kw = ["service","services","support","maintenance","consulting","subscription"]
    prod_kw = ["sku","model","part","item","serial","unit","product"]
    if any(k in t for k in svc_kw) and not any(k in t for k in prod_kw):
        return "service"
    if any(k in t for k in prod_kw) and not any(k in t for k in svc_kw):
        return "product"
    return None

def tag_field(name: str, value: str) -> Dict[str, Any]:
    tags: List[str] = []
    norm: Dict[str, Any] = {}

    n_low = (name or "").lower()
    v = (value or "").strip()

    # direct hints from field name
    if any(k in n_low for k in ["email","e-mail"]): tags.append("email")
    if "address" in n_low: tags.append("address")
    if any(k in n_low for k in ["ip","ip address"]): tags.append("ip_address")
    if any(k in n_low for k in ["amount","total","balance","price","fee"]): tags.append("price")
    if any(k in n_low for k in ["date","due","invoice date","statement date"]): tags.append("date")
    if any(k in n_low for k in ["name","customer","payee","recipient","authorized signer"]): tags.append("person_or_org")

    # value-based detection
    if is_email(v): tags.append("email")
    if is_ip(v): tags.append("ip_address")
    ok_date, iso = is_date(v)
    if ok_date:
        tags.append("date")
        if iso: norm["date_iso"] = iso
    if is_money(v):
        tags.append("price")
        nm = norm_money(v)
        if nm is not None: norm["amount"] = nm
    if is_us_address(v): tags.append("address")

    # entity-level
    if looks_like_org(v): tags.append("organization")
    elif looks_like_person(v): tags.append("person")

    # product/service heuristic
    ps = product_or_service(v)
    if ps: tags.append(ps)

    # canonical supertype
    if "organization" in tags or "person" in tags or "person_or_org" in tags:
        tags.append("entity")

    # dedupe and order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    return {"semantics": tags, "normalized": (norm if norm else None)}
