
from typing import List
from .schemas import DocumentResult, OCRWord
from .ocr import extract_words
from .extractors.invoice import heuristic_invoice_extract
from .extractors.statement import heuristic_statement_extract
from .validation import validate_invoice, validate_statement

def classify_doc(words: List[OCRWord]) -> str:
    joined = " ".join(w.text.lower() for w in words)
    if "invoice" in joined: return "invoice"
    if "statement" in joined or "opening balance" in joined: return "bank_statement"
    return "unknown"

def process_document(path: str) -> DocumentResult:
    words, size = extract_words(path)
    doc_type = classify_doc(words)
    fields, tables, issues = [], [], []
    if doc_type == "invoice":
        fields, tables = heuristic_invoice_extract(words)
        issues.extend(validate_invoice(DocumentResult(doc_type=doc_type, pages=0, words=words, fields=fields, tables=tables)))
    elif doc_type == "bank_statement":
        fields, tables = heuristic_statement_extract(words)
        issues.extend(validate_statement(DocumentResult(doc_type=doc_type, pages=0, words=words, fields=fields, tables=tables)))
    else:
        f1,t1 = heuristic_invoice_extract(words)
        f2,t2 = heuristic_statement_extract(words)
        if len(f1)+len(t1) >= len(f2)+len(t2):
            doc_type, fields, tables = "invoice?", f1, t1
        else:
            doc_type, fields, tables = "bank_statement?", f2, t2
    doc = DocumentResult(
        doc_type=doc_type, pages=0, words=words, fields=fields, tables=tables, issues=issues,
        meta={"page_size":{"w": size[0], "h": size[1]}}
    )


# --- semantic tagging step ---
from .semantics import tag_field
def apply_semantics(doc: DocumentResult) -> DocumentResult:
    # Tag fields
    for f in doc.fields:
        tagged = tag_field(f.name, f.value)
        f.semantics = tagged.get("semantics", [])
        f.normalized = tagged.get("normalized")
    # Tag table headers/cells lightly (only detect price/date/product/service per cell text)
    from .semantics import is_money, is_date, product_or_service
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                tags = []
                if is_money(c.text): tags.append("price")
                ok, iso = is_date(c.text)
                if ok: tags.append("date")
                ps = product_or_service(c.text)
                if ps: tags.append(ps)
                if tags and (not hasattr(c, "semantics")):
                    # annotate via ad-hoc attributes (not in schema, but safe for UI read-only)
                    setattr(c, "semantics", tags)
    return doc
