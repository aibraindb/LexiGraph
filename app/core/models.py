from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1 (PDF coords)

@dataclass
class Line:
    text: str
    bbox: BBox

@dataclass
class Block:
    bbox: BBox
    lines: List[Line]

@dataclass
class Page:
    number: int       # 1-based
    width: int
    height: int
    blocks: List[Block]

@dataclass
class OCRIndex:
    pages: List[Page]
    images: Dict[int, str] = field(default_factory=dict)  # page_num -> PNG (base64)

@dataclass
class Edit:
    field_id: str
    page: int
    bbox: BBox
    method: str  # 'move' | 'draw' | 'autolink'
    value: Optional[str] = None
    confidence: Optional[float] = None

def todict(obj):  # tiny helper for JSON dumps
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [todict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: todict(v) for k, v in obj.items()}
    return obj
