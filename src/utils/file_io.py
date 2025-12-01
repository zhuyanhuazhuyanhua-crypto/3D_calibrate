import os
from pathlib import Path
import yaml

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_yaml(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(obj, p):
    ensure_dir(os.path.dirname(p))
    with open(p, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, allow_unicode=True)
