import logging
import os
import time

# config
try:
    # inside try to be able to easily run stuff on ipython
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
except NameError:
    BASE_DIR = "."

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PARAMS_FILENAME = "params.yml"
LOG_FILENAME = "log.log"


def configure_logger(output_file: str = LOG_FILENAME):
    """Configure logger with console logging and persistent logging handler.

    Parameters
    ----------
    output_file : str, optional
        file to store logs, by default LOG_FILENAME

    Returns
    -------
    logger object
    """
    logfmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    # console handler
    c_handler = logging.StreamHandler()
    # file handler
    f_handler = logging.FileHandler(output_file)

    log_handlers = [c_handler, f_handler]

    # Create formatters and add it to handlers
    formatter = logging.Formatter(fmt=logfmt, datefmt=datefmt)

    for handler in log_handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set level
    logger.setLevel(logging.DEBUG)
    return logger


def generate_filename() -> str:
    """Create unique filename for both the logfile and model related output.

    Returns
    -------
    str
        filename
    """
    return time.strftime("%Y%m%d-%H%M%S")


logs_dir = os.path.join(LOGS_DIR, "logs_" + generate_filename())
# logs_dir = os.path.join(BASE_DIR, "static_output")
if not os.path.exists(logs_dir):
    try:
        os.makedirs(logs_dir)
    except FileExistsError:
        pass

logger = configure_logger(output_file=os.path.join(logs_dir, LOG_FILENAME))
logger.propagate = False
