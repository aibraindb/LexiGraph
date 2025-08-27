
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    page: int = Field(..., description="0-based page index")
    x0: float; y0: float; x1: float; y1: float

class OCRWord(BaseModel):
    text: str
    bbox: BoundingBox
    confidence: Optional[float] = None

class ExtractedField(BaseModel):
    name: str; value: str
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    issues: List[str] = []
    semantics: List[str] = []
    normalized: Optional[Dict[str, Any]] = None

class TableCell(BaseModel):
    text: str
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None

class TableRow(BaseModel):
    cells: List[TableCell]

class ExtractedTable(BaseModel):
    name: str
    rows: List[TableRow] = []
    header: Optional[List[str]] = None

class DocumentResult(BaseModel):
    doc_type: str
    pages: int
    words: List[OCRWord]
    fields: List[ExtractedField] = []
    tables: List[ExtractedTable] = []
    issues: List[str] = []
    semantics: List[str] = []
    normalized: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = {}
