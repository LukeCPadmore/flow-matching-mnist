import logging
import tempfile
import os
from typing import Tuple
from tqdm import tqdm

class TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_temp_logger(name: str = "training") -> Tuple[logging.Logger, str]:
    """
    Create a logger that logs to stdout and to a temporary file.
    Returns (logger, log_path).
    """
    # Per-call temp dir, you can clean this up after logging to MLflow
    tmpdir = tempfile.mkdtemp(prefix=f"{name}_")
    log_path = os.path.join(tmpdir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate into root logger

    # Clear existing handlers in case this is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = TqdmStreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger, log_path
