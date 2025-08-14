# LexiGraph-Simple-v3

Upload → Embed → Link to FIBO. No rule engine.

## Run
API:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```
UI:
```
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
```
