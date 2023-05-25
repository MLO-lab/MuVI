"""Multi-view LVM with informed structured sparsity."""

import logging
import logging.config
from pathlib import Path

from .core import *  # noqa: F401, F403
from .tools import external as ext
from .tools import feature_sets as fs
from .tools import load
from .tools import plotting as pl
from .tools import save
from .tools import utils as tl

logging.config.fileConfig(
    Path(__file__).resolve().parent / "log.conf",
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"

__all__ = ["fs", "pl", "tl", "ext", "save", "load"]
