# LexiGraph — WF‑ECM overlay (HITL demo)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-wf.txt
# Optional: put your full FIBO at data/fibo_full.ttl (not required for WF overlay)
python -c "from app.core.wf_vec import build_wf_index; print(build_wf_index())"
streamlit run ui/streamlit_wf.py
