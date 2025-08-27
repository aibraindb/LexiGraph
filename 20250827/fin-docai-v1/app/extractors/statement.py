
import re
from typing import List, Tuple
from ..schemas import OCRWord, ExtractedField, ExtractedTable, TableRow, TableCell

def heuristic_statement_extract(words: List[OCRWord]) -> Tuple[List[ExtractedField], List[ExtractedTable]]:
    fields: List[ExtractedField] = []
    tables: List[ExtractedTable] = []

    joined = " ".join(w.text for w in words)
    acct = re.search(r"account\s*(number|no\.)?[:#]?\s*([A-Za-z0-9-]{4,})", joined, re.I)
    if acct: fields.append(ExtractedField(name="Account Number", value=acct.group(2)))
    opening = re.search(r"opening\s*balance[:#]?\s*\$?([0-9,]+\.\d{2})", joined, re.I)
    closing = re.search(r"closing\s*balance[:#]?\s*\$?([0-9,]+\.\d{2})", joined, re.I)
    if opening: fields.append(ExtractedField(name="Opening Balance", value=opening.group(1)))
    if closing: fields.append(ExtractedField(name="Closing Balance", value=closing.group(1)))

    table = ExtractedTable(name="Transactions", header=["Date","Description","Amount","Balance"])
    currency = re.compile(r"\$?[0-9,]+\.\d{2}")
    datepat = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
    lines = {}
    for w in words: lines.setdefault(round(w.bbox.y0/5.0), []).append(w.text)
    for y, toks in sorted(lines.items()):
        dates = [t for t in toks if datepat.fullmatch(t)]
        amts  = [t for t in toks if currency.fullmatch(t)]
        if dates and amts:
            date = dates[0]; amount = amts[0]; balance = amts[1] if len(amts)>1 else ""
            desc = " ".join([t for t in toks if not datepat.fullmatch(t) and not currency.fullmatch(t)])
            table.rows.append(TableRow(cells=[
                TableCell(text=date), TableCell(text=desc), TableCell(text=amount), TableCell(text=balance)
            ]))
    if table.rows: tables.append(table)
    return fields, tables
