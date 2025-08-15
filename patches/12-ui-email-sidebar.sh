#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("ui/streamlit_app.py")
s = p.read_text()

# Inject a sidebar email block under API settings in sidebar
needle = "st.markdown(\"### 🧭 Context Association (CA)\")"
if needle in s and "Email Context (Sidebar)" not in s:
    insert = """
    st.markdown("---")
    st.markdown("### 📧 Email Context (Sidebar)")
    if st.button("Load threads (sidebar)"):
        try:
            r = requests.get(f"{api_base}/demo/threads", timeout=10); r.raise_for_status()
            st.session_state["threads"] = r.json()
        except Exception as e:
            st.error(e)
    threads = st.session_state.get("threads", [])
    if threads:
        opt = st.selectbox("Choose email", [f"{t['id']} — {t['subject']}" for t in threads], key="sb_email_pick")
        if opt:
            th = next((t for t in threads if opt.startswith(t['id'])), None)
            if th:
                st.caption(th["body"])
                b1,b2,b3 = st.columns(3)
                if b1.button("Bind CA", key="sb_bind"):
                    st.session_state["case_id"] = f"CASE-{th['id']}"
                    st.session_state["product_id"] = "LOAN" if "Loan" in th["subject"] else "LEASE"
                    st.session_state["customer_id"] = "CUST-ALPHA"
                    try:
                        r = requests.post(f"{api_base}/ca/associate",
                                          json={"case_id": st.session_state["case_id"],
                                                "product_id": st.session_state["product_id"],
                                                "customer_id": st.session_state["customer_id"]},
                                          timeout=15)
                        r.raise_for_status()
                        st.session_state["ca"] = r.json()
                        st.success(f"CA bound: {st.session_state['case_id']}")
                    except Exception as e:
                        st.error(e)
                if b2.button("Prime", key="sb_prime"):
                    try:
                        pr = requests.post(f"{api_base}/demo/prime",
                                           json={"case_id": st.session_state.get("case_id","CASE-ALPHA"),
                                                 "email_id": th["id"]},
                                           timeout=20); pr.raise_for_status()
                        st.success(f"Primed: {pr.json().get('primed_doc_id')}")
                    except Exception as e:
                        st.error(e)
                if b3.button("Ingest", key="sb_ingest"):
                    try:
                        ir = requests.post(f"{api_base}/demo/ingest",
                                           json={"case_id": st.session_state.get("case_id","CASE-ALPHA"),
                                                 "email_id": th["id"]},
                                           timeout=60); ir.raise_for_status()
                        st.info("Ingested attachments and updated CA.")
                    except Exception as e:
                        st.error(e)
    """
    s = s.replace(needle, needle + insert)

Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: Email simulator moved to sidebar."
