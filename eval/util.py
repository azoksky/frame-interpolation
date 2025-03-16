import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTHONWARNINGS'] = 'ignore'
import shutil
from typing import Generator, Iterable, List, Optional

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
import logging

# Set up a logger for this module.
logger = logging.getLogger("frame_interpolation.util")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def read_image(filename: str) -> np.ndarray:
    """Reads an sRGB 8-bit image from disk and returns it as a float32 array with values in [0,1]."""
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 image (values in [0,1]) as a PNG or JPEG file."""
    image_in_uint8 = (np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.jpg':
        image_data = tf.io.encode_jpeg(image_in_uint8)
    else:
        image_data = tf.io.encode_png(image_in_uint8)
    tf.io.write_file(filename, image_data)

def _recursive_generator(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    interpolator: interpolator_lib.Interpolator,
    total: int,
    progress: dict,
    gpu_info: str
) -> Generator[np.ndarray, None, None]:
    """Recursively generates the in-between frames between two input images."""
    if num_recursions == 0:
        yield frame1
    else:
        time_val = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(frame1[np.newaxis, ...], frame2[np.newaxis, ...], time_val)[0]
        progress["count"] += 1
        if progress["count"] % 10 == 0 or progress["count"] == total:
            logger.info("GPU %s: Processed %d/%d interpolation steps", gpu_info, progress["count"], total)
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                        interpolator, total, progress, gpu_info)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                        interpolator, total, progress, gpu_info)

def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator,
    gpu_info: Optional[str] = None
) -> Iterable[np.ndarray]:
    """
    Given a list of file paths, reads them and recursively interpolates between each adjacent pair.
    Returns a generator yielding all the frames (both original and interpolated).
    """
    n = len(frames)
    total = (n - 1) * (2**times_to_interpolate - 1)
    if gpu_info is None:
        gpu_info = tf.test.gpu_device_name() or "CPU"
    logger.info("Starting recursive interpolation on GPU: %s with total steps: %d", gpu_info, total)
    progress = {"count": 0}
    for i in range(1, n):
        yield from _recursive_generator(read_image(frames[i - 1]), read_image(frames[i]),
                                        times_to_interpolate, interpolator, total, progress, gpu_info)
    yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator,
    gpu_info: Optional[str] = None
) -> Iterable[np.ndarray]:
    """
    Same as interpolate_recursively_from_files() but for frames already loaded in memory.
    """
    n = len(frames)
    total = (n - 1) * (2**times_to_interpolate - 1)
    if gpu_info is None:
        gpu_info = tf.test.gpu_device_name() or "CPU"
    logger.info("Starting recursive interpolation on GPU: %s with total steps: %d", gpu_info, total)
    progress = {"count": 0}
    for i in range(1, n):
        yield from _recursive_generator(frames[i - 1], frames[i],
                                        times_to_interpolate, interpolator, total, progress, gpu_info)
    yield frames[-1]

