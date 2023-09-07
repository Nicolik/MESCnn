import os
from definitions import ROOT_DIR


def get_logs_path(root_dir=None):
    root_dir = ROOT_DIR if root_dir is None else root_dir
    logs_dir = os.path.join(root_dir, 'classification', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir
