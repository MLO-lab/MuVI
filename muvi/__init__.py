"""Multi-view LVM with informed structured sparsity."""

import logging
import logging.config

from pathlib import Path

from muvi.core import *  # noqa: F403
from muvi.tools import external as ext
from muvi.tools import feature_sets as fs
from muvi.tools import load
from muvi.tools import plotting as pl
from muvi.tools import save
from muvi.tools import utils as tl


logging.config.fileConfig(
    Path(__file__).resolve().parent / "log.conf",
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__all__ = ["fs", "pl", "tl", "ext", "save", "load"]
