from __future__ import annotations
from typing import List, Dict, Any
import re

NUM_RX  = re.compile(r'(^|\b)(?:USD\s*)?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)\b', re.I)
DATE_RX = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b')
PCT_RX  = re.compile(r'\b\d{1,2}(?:\.\d+)?\s*%\b')

def _clean(t: str) -> str:
    return re.sub(r'\s+', ' ', (t or '')).strip()

def ocr_to_lines(ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ocr_items: return []
    items = sorted(ocr_items, key=lambda o: (o['bbox'][1], o['bbox'][0]))
    lines, cur, cur_y = [], [], items[0]['bbox'][1]
    for it in items:
        y = it['bbox'][1]
        if abs(y - cur_y) > 12:
            if cur:
                text = ' '.join(_clean(e['text']) for e in cur if e['text'])
                bbox = [min(e['bbox'][0] for e in cur), min(e['bbox'][1] for e in cur),
                        max(e['bbox'][2] for e in cur), max(e['bbox'][3] for e in cur)]
                lines.append({'text': text, 'bbox': bbox, 'n': len(cur)})
            cur, cur_y = [it], y
        else:
            cur.append(it)
    if cur:
        text = ' '.join(_clean(e['text']) for e in cur if e['text'])
        bbox = [min(e['bbox'][0] for e in cur), min(e['bbox'][1] for e in cur),
                max(e['bbox'][2] for e in cur), max(e['bbox'][3] for e in cur)]
        lines.append({'text': text, 'bbox': bbox, 'n': len(cur)})
    return lines

def kv_from_lines(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    kv: Dict[str, Any] = {}
    for ln in lines:
        txt = _clean(ln['text'])
        if ':' in txt:
            k, v = [p.strip() for p in txt.split(':', 1)]
            if k and v:
                kv[k.lower()] = v
    cands = []
    for ln in lines:
        txt = _clean(ln['text'])
        score = 0
        if NUM_RX.search(txt):  score += 2
        if DATE_RX.search(txt): score += 2
        if PCT_RX.search(txt):  score += 1
        if score: cands.append({'text': txt, 'bbox': ln['bbox']})
    for c in cands:
        cx, cy = ((c['bbox'][0]+c['bbox'][2])/2, (c['bbox'][1]+c['bbox'][3])/2)
        best = None
        for ln in lines:
            t = _clean(ln['text'])
            if len(t.split()) <= 3 and len(t) <= 24 and t and t[-1] != ':':
                lx, ly = ((ln['bbox'][0]+ln['bbox'][2])/2, (ln['bbox'][1]+ln['bbox'][3])/2)
                if lx <= cx + 5 and ly <= cy + 40:
                    d = ((lx-cx)**2 + (ly-cy)**2) ** 0.5 - 0.4*(cx-lx)
                    if best is None or d < best[0]:
                        best = (d, t.lower())
        if best:
            kv.setdefault(best[1], c['text'])
    return kv
