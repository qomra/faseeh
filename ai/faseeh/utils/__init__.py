import os
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def save_yaml(data, file_path):
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def full_or_augment(path,root_path):
    if not os.path.isabs(path):
        return os.path.join(root_path,path)
    return path