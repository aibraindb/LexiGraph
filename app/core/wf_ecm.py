from rdflib import Graph, RDFS, OWL, URIRef, Namespace
from rdflib.util import guess_format
from pathlib import Path
import yaml, re

WF = Namespace("https://wells.local/ecm#")

DATA = Path("data")
CFG  = Path("config")
WF_TTL = DATA/"wf_ecm.ttl"
WF_MAP = CFG/"wf_mappings.yaml"

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

class WFOnto:
    def __init__(self):
        if not WF_TTL.exists():
            raise FileNotFoundError("Missing data/wf_ecm.ttl")
        self.g = Graph()
        self.g.parse(WF_TTL, format=guess_format(str(WF_TTL)) or "turtle")
        self.cfg = yaml.safe_load(WF_MAP.read_text()) if WF_MAP.exists() else {"classes":{}}

    def classes(self):
        out = []
        # Any subject that is declared as Class in our namespace
        for s,_,_ in self.g.triples((None, None, None)):
            if isinstance(s, URIRef) and str(s).startswith(str(WF)):
                if (s, None, OWL.Class) in self.g or (s, None, RDFS.Class) in self.g:
                    lbl = self.g.value(s, RDFS.label)
                    out.append({"uri": str(s), "label": str(lbl) if lbl else s.split("#")[-1]})
        uniq = {c["uri"]: c for c in out}
        return list(uniq.values())

    def props_for(self, class_uri: str):
        cu = URIRef(class_uri)
        props=[]
        for p,_,dom in self.g.triples((None, RDFS.domain, None)):
            if dom == cu:
                lbl = self.g.value(p, RDFS.label)
                row = {"property": str(p), "label": (str(lbl) if lbl else str(p).split("#")[-1])}
                # overlay aliases/types from config
                q = self._qname(str(p))
                cls_cfg = self.cfg.get("classes",{}).get(self._qname(class_uri), {})
                prop_cfg = (cls_cfg.get("properties") or {}).get(q, {})
                row["aliases"] = prop_cfg.get("aliases", [])
                row["type"] = prop_cfg.get("type", "string")
                row["fibo_equiv"] = prop_cfg.get("fibo_equiv")
                props.append(row)
        props.sort(key=lambda x: x["label"].lower())
        return props

    def search(self, q: str, limit=20):
        ql = q.strip().lower()
        if not ql: return []
        hits=[]
        for c in self.classes():
            hay = (c["label"]+" "+c["uri"].split("#")[-1]+" "+_CAMEL_RE.sub(" ", c["uri"].split("#")[-1])).lower()
            if ql in hay:
                hits.append(c)
                if len(hits)>=limit: break
        return hits

    def _qname(self, uri: str) -> str:
        if uri.startswith(str(WF)): return "wf:" + uri.split("#")[-1]
        return uri

WF_ONTO = None
def get_wf():
    global WF_ONTO
    if WF_ONTO is None:
        WF_ONTO = WFOnto()
    return WF_ONTO
