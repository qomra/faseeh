import os
from pathlib import Path
from ruamel.yaml import YAML

def load_yaml(file_path):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # Prevent line wrapping
    yaml.default_flow_style = False
    with Path(file_path).open('r') as f:
        return yaml.load(f)
    
def save_yaml(data, file_path):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.default_flow_style = False
    with Path(file_path).open('w') as f:
        yaml.dump(data, f)

def full_or_augment(path,root_path):
    if not os.path.isabs(path):
        return os.path.join(root_path,path)
    return path
