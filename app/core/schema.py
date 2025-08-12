from typing import Dict, Any
import yaml, os, copy, os.path as p

def _load_yaml(path:str): 
    with open(path,'r') as f: 
        return yaml.safe_load(f)

def load_canonical(config_dir:str)->Dict[str,Any]:
    return _load_yaml(p.join(config_dir,'field_dictionary.yaml'))

def load_variant(config_dir:str, variant_id:str)->Dict[str,Any]:
    vdir = p.join(config_dir,'variants')
    for name in os.listdir(vdir):
        if not name.endswith('.yaml'): continue
        obj = _load_yaml(p.join(vdir,name))
        if obj.get('variant_id')==variant_id: return obj
    raise FileNotFoundError(variant_id)

def resolve_effective_schema(config_dir:str, variant_id:str)->Dict[str,Any]:
    canonical = load_canonical(config_dir)
    variant = load_variant(config_dir, variant_id)
    eff = {"variant_id":variant_id,"fields":{}}
    vfields = variant.get('extract',{}).get('fields',{})
    for fname, vcfg in vfields.items():
        base = canonical['fields'].get(fname,{})
        merged = copy.deepcopy(base); merged.update(vcfg)
        eff["fields"][fname] = merged
    return eff
