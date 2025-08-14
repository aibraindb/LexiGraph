import json
from pathlib import Path

MAP_PATH = Path("config/fibo_field_map.json")

def load_schema_map():
    try:
        return json.loads(MAP_PATH.read_text())
    except Exception:
        return {}

def suggest_schema_for(label: str):
    m = load_schema_map()
    return m.get(label)
