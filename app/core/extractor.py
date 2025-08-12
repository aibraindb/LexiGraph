from typing import Dict, Any, List
import re
from .utils import parse_money, parse_date

def _compile(patterns: List[str]):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

def run_regex(text:str, cfg:Dict[str,Any]):
    for pat in _compile(cfg.get('patterns',[])):
        m = pat.search(text)
        if not m: continue
        groups = [g for g in m.groups() if g]
        val = groups[-1] if groups else m.group(0)
        if 'date' in cfg.get('name',''):
            val = parse_date(val) or val
        if 'amount' in cfg.get('name','') or 'total' in cfg.get('name',''):
            val = parse_money(val) or val
        return {"value": val.strip(), "confidence": 0.85, "strategy":"regex", "evidence":{"pattern":pat.pattern}}
    return {"value": None, "confidence": 0.0, "strategy":"regex", "evidence":{"reason":"no match"}}

def run_literal(cfg:Dict[str,Any]):
    return {"value": cfg.get('value'), "confidence": 0.99, "strategy":"literal", "evidence":{"reason":"literal"}}

def run_anchor_window(text:str, cfg:Dict[str,Any]):
    anchor = cfg.get('anchor','')
    lines = text.splitlines()
    for i,line in enumerate(lines):
        if anchor.lower() in line.lower():
            j = min(i + int(cfg.get('window_lines_after',1)), len(lines)-1)
            val = lines[j].strip()
            return {"value": val, "confidence": 0.70, "strategy":"anchor_window", "evidence":{"anchor":anchor,"line":i,"picked":j}}
    return {"value": None, "confidence": 0.0, "strategy":"anchor_window", "evidence":{"reason":"anchor not found","anchor":anchor}}

def extract_fields(text:str, effective_schema:Dict[str,Any])->Dict[str,Any]:
    out = {}
    for fname, cfg in effective_schema['fields'].items():
        cfg = dict(cfg); cfg['name']=fname
        strat = cfg.get('strategy')
        if strat=='regex':
            out[fname] = run_regex(text, cfg)
        elif strat=='anchor_window':
            out[fname] = run_anchor_window(text, cfg)
        elif strat=='literal':
            out[fname] = run_literal(cfg)
        else:
            out[fname] = {"value": None, "confidence": 0.0, "strategy":strat or "unknown", "evidence":{"reason":"unknown strategy"}}
    return out
