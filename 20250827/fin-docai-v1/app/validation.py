
from typing import List
from .schemas import DocumentResult

def _to_amount(s: str) -> float:
    s = s.replace("$","").replace(",","").strip()
    try: return float(s)
    except: return 0.0

def validate_invoice(doc: DocumentResult) -> List[str]:
    issues: List[str] = []
    total_field = next((f for f in doc.fields if f.name.lower()=="total"), None)
    items = next((t for t in doc.tables if t.name=="LineItems"), None)
    if total_field and items and items.rows:
        s = 0.0
        for r in items.rows:
            if len(r.cells)>=4: s += _to_amount(r.cells[-1].text)
        if abs(s - _to_amount(total_field.value)) > 0.01:
            issues.append(f"Invoice line-item sum {s:.2f} != Total {_to_amount(total_field.value):.2f}")
    else:
        issues.append("Invoice validation: missing total or items")
    return issues

def validate_statement(doc: DocumentResult) -> List[str]:
    issues: List[str] = []
    opening = next((f for f in doc.fields if f.name.lower()=="opening balance"), None)
    closing = next((f for f in doc.fields if f.name.lower()=="closing balance"), None)
    tx = next((t for t in doc.tables if t.name=="Transactions"), None)
    if opening and closing and tx:
        net = 0.0
        for r in tx.rows:
            if len(r.cells)>=3: net += _to_amount(r.cells[2].text)
        if abs((_to_amount(opening.value)+net) - _to_amount(closing.value)) > 0.01:
            issues.append("Statement reconciliation failed (opening + net != closing)")
    else:
        issues.append("Statement validation: missing balances or transactions")
    return issues
