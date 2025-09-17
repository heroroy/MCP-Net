"""
utils.py

Helper functions for reproducibility, metrics visualization, etc.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seeds(seed=42):
    """
    Fixes random seeds for reproducibility.
    Args:
        seed: Seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
