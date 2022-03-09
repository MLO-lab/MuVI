import os

try:
    # inside try to be able to easily run stuff on ipython
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
except NameError:
    BASE_DIR = "."

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
