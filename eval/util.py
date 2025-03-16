import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTHONWARNINGS'] = 'ignore'
import shutil
from typing import Generator, Iterable, List

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
import logging
from tqdm.auto import tqdm

# Fallback for TqdmLoggingHandler in case it's not available.
try:
    from tqdm.contrib.logging import TqdmLoggingHandler
except ImportError:
    class TqdmLoggingHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

logger = logging.getLogger("frame_interpolation.util")
if not logger.handlers:
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger
