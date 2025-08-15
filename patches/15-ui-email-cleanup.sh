#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path, re
p = Path("ui/streamlit_app.py")
s = p.read_text()

# 1) Remove / neutralize mid-page "0.5 Email Context Simulator" block
s = re.sub(r"st\.header\(\"0\.5\).*?st\.write\(", "st.write(", s, flags=re.S)

# 2) Add Sidebar Email App (if not present)
if "### 📬 Email Inbox" not in s:
    insert_after = "st.markdown(\"### 🧭 Context Association (CA)\")"
    block = '''
    st.markdown("---")
    st.markdown("### 📬 Email Inbox")
    try:
        r = requests.get(f"{api_base}/demo/threads", timeout=10); r.raise_for_status()
        threads = r.json()
    except Exception:
        threads = []

    if threads:
        sel = st.selectbox("Select email", [f"{t['id']} — {t['subject']}" for t in threads], key="email_pick")
        th = next((t for t in threads if sel and sel.startswith(t['id'])), None)
        if th:
            # Auto-CA bind silently
            try:
                _ = requests.post(f"{api_base}/ca/associate",
                                  json={"case_id": f"CASE-{th['id']}",
                                        "product_id": ("LOAN" if "Loan" in th["subject"] else "LEASE"),
                                        "customer_id": "CUST-ALPHA"},
                                  timeout=10)
            except Exception:
                pass
            st.caption(th["body"])
            st.markdown("**Attachments**")
            # Simulate two attachments; user picks local file to upload
            for i, att in enumerate(["attachment-1.pdf", "attachment-2.pdf"], start=1):
                st.write(f"📎 {att}")
                up = st.file_uploader(f"Attach local file for {att}", type=["pdf","txt"], key=f"att_{th['id']}_{i}")
                if up:
                    try:
                        r = requests.post(f"{api_base}/upload",
                                          files={"file":(up.name, up.getbuffer(), "application/octet-stream")},
                                          timeout=120)
                        r.raise_for_status()
                        res = r.json()
                        st.success(f"Uploaded as doc_id {res['doc_id']}")
                    except Exception as e:
                        st.error(e)
    '''
    s = s.replace(insert_after, insert_after + block)

Path(p).write_text(s)
print("Patched:", p)
PY

echo "UI email cleanup done."
