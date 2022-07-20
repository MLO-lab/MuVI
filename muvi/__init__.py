"""Multi-view LVM with informed structured sparsity."""

import logging
import logging.config
import os

from .core import *  # noqa: F401, F403
from .tools import config as cfg
from .tools import feature_sets as fs
from .tools import plotting as pl
from .tools import utils as tl

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.conf"),
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"

__all__ = ["cfg", "fs", "pl", "tl"]
