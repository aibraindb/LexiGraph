# LexiGraph v3 — Document Intelligence Platform

Enhancements in this build:
- Train-on-upload (few-shot) + variant dropdown
- Variant wizard (suggest anchors/fields from a doc; create YAML; hot-reload)
- Undo last training (`/index/undo_last`) + delete by id
- Rebuild index from dataset (`/index/rebuild`)
- Bigger FIBO TTL loader (UI; plus `scripts/fibo_fetch.py`)
- Confusion/anchors report (`GET /report/anchors?label_a=...&label_b=...`)
- Better anchor-window extraction heuristics

## Run
API:
```
cd lexigraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

UI:
```
cd lexigraph
source .venv/bin/activate
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
```

## Useful endpoints
- `POST /upload` → (auto/manual) label, save to dataset, train now (optional)
- `POST /train/label` → label & persist embeddings
- `GET /variants/list` → variants
- `POST /variants/suggest` → propose anchors/fields from a doc
- `POST /variants/create` → write YAML + hot-reload
- `GET /index/stats`, `GET /index/list`, `DELETE /index/delete`, `POST /index/undo_last`
- `GET /dataset/stats`, `POST /index/rebuild`
- `GET /report/anchors` → anchor suggestions (positives/negatives)

## Data locations
- Embeddings: `lexigraph/data/vector_index/`
- Dataset texts: `lexigraph/data/dataset/<variant>/*.txt`
- Rule-mining copies: `lexigraph/data/labeled/<variant>/*.txt`
