
import re
from typing import List, Tuple
from ..schemas import OCRWord, ExtractedField, ExtractedTable, TableRow, TableCell

def heuristic_invoice_extract(words: List[OCRWord]) -> Tuple[List[ExtractedField], List[ExtractedTable]]:
    fields: List[ExtractedField] = []
    tables: List[ExtractedTable] = []

    candidates = {
        "Invoice Number": r"(invoice\s*(no\.|number)?[:#]?\s*)([A-Za-z0-9-]+)",
        "Invoice Date":   r"(invoice\s*date[:#]?\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        "Due Date":       r"(due\s*date[:#]?\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        "Total":          r"(total\s*(due)?[:#]?\s*\$?)([0-9,]+\.\d{2})",
    }
    joined = " ".join(w.text for w in words)
    for name, pat in candidates.items():
        m = re.search(pat, joined, re.I)
        if m:
            fields.append(ExtractedField(name=name, value=m.group(m.lastindex) if m.lastindex else m.group(0)))

    currency = re.compile(r"\$?[0-9,]+\.\d{2}")
    lines = {}
    for w in words: lines.setdefault(round(w.bbox.y0/5.0), []).append(w.text)
    table = ExtractedTable(name="LineItems", header=["Description","Qty","Price","Amount"])
    for y, toks in sorted(lines.items()):
        amts = [t for t in toks if currency.fullmatch(t)]
        if len(amts) >= 2 and any(len(t)>2 and t.isalpha() for t in toks):
            qty = next((t for t in toks if re.fullmatch(r"\d+", t)), "1")
            price = amts[-2] if len(amts) >= 2 else ""
            amount = amts[-1] if amts else ""
            desc = " ".join([t for t in toks if not re.fullmatch(r"\d+",t) and not currency.fullmatch(t)])
            table.rows.append(TableRow(cells=[
                TableCell(text=desc), TableCell(text=qty), TableCell(text=price), TableCell(text=amount)
            ]))
    if table.rows: tables.append(table)
    return fields, tables
