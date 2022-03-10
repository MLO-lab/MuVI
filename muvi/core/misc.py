import os
import time

import numpy as np


def generate_filename() -> str:
    """Create unique filename"""
    return time.strftime("%Y%m%d-%H%M%S")


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)
