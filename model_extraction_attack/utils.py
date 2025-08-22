"""Utility functions for setting up experiments, logging, and environment snapshots."""

from __future__ import annotations
import json, random, platform, sys
import numpy as np
import torch, logging
from torch import nn # Added for _soft_cross_entropy

def set_seed(seed: int=42):
    """Sets the random seed for reproducibility across different libraries."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup_logger(log: bool=True, level: str="INFO"):
    """Sets up the logging configuration for the experiment."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger()

def save_json(path, obj):
    """Saves a Python object to a JSON file."""
    def _conv(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
        except Exception: pass
        return o
    path.write_text(json.dumps(obj, default=_conv, indent=2), encoding="utf-8")

def snapshot_env(path):
    """Captures and saves a snapshot of the current Python environment and installed packages."""
    info = {}
    try:
        import torch, torchvision, numpy, PIL, sklearn
        info.update({
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "sklearn": sklearn.__version__,
        })
    except Exception:
        pass
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")

def soft_cross_entropy(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    """Calculates soft cross-entropy loss."""
    return -torch.mean(torch.sum(y_soft * nn.functional.log_softmax(logits, dim=1), dim=1))