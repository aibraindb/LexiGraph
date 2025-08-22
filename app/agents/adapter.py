# app/agents/adapter.py
from typing import Dict, Any, List

def run_ca_pipeline(email_threads: List[Dict[str, Any]], attachments: List[bytes]) -> Dict[str, Any]:
    """
    Input:
      - email_threads: list of {subject, sender, recipients, body, date, attachments_meta}
      - attachments: raw bytes for doc uploads (optional)
    Output:
      - context: inferred {customer_id, product_id, case_id, doc_requirements, prior_docs}
    """
    # TODAY: stub out with simple heuristic
    ctx = {"customer_id":"CUST-ALPHA","product_id":"LOAN","case_id":"CASE-ALPHA",
           "doc_requirements":["W2","401k","BankStatement","LoanAgreement"]}
    return ctx

def run_schema_proposal(doc_text: str, fibo_hits: List[Dict[str,Any]]) -> Dict[str, Any]:
    """
    Decide target attributes (merge of multiple FIBO classes if needed).
    """
    # TODAY: pick top hit and return starter fields
    props = [{"name":"borrowerName"}, {"name":"principalAmount"}, {"name":"effectiveDate"}]
    return {"fibo_class": fibo_hits[0]["uri"] if fibo_hits else None, "attributes": props}
