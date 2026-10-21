import json
import os
import platform
import random
import sys

import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_versions(path="results/software_versions.json"):
    """Record software versions next to the results."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ["numpy", "scipy", "sklearn", "mne", "torch", "pandas", "pyriemann"]:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(versions, f, indent=2)
    return versions