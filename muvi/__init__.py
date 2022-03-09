"""Multi-view LVM with informed structured sparsity."""
import logging
import logging.config
import os

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log.conf")
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"
