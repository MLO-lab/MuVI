import os
import time

try:
    # inside try to be able to easily run stuff on ipython
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
except NameError:
    BASE_DIR = "."

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def generate_filename() -> str:
    """Create unique filename"""
    return time.strftime("%Y%m%d-%H%M%S")
