import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTHONWARNINGS'] = 'ignore'
import shutil
from typing import Generator, Iterable, List, Optional

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
import logging
from tqdm.auto import tqdm

# Set up logger.
logger = logging.getLogger("frame_interpolation.util")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'

def read_image(filename: str) -> np.ndarray:
    """Reads an sRgb 8-bit image and returns a float32 array in [0,1]."""
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 image (values in [0,1]) to a file (PNG or JPEG)."""
    image_in_uint8 = np.clip(image * _UINT8_MAX_F, 0, _UINT8_MAX_F).astype(np.uint8)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        encoded = tf.io.encode_jpeg(image_in_uint8)
    else:
        encoded = tf.io.encode_png(image_in_uint8)
    tf.io.write_file(filename, encoded)

def get_valid_device() -> str:
    """Returns a valid GPU device string (e.g. '/device:GPU:0') or 'CPU'."""
    device = tf.test.gpu_device_name() or "CPU"
    if device.startswith("/physical_device:"):
        device = device.replace("/physical_device", "/device")
    return device

def _recursive_generator(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    interpolator: interpolator_lib.Interpolator,
    total: int,
    progress: dict,
    gpu_info: str
) -> Generator[np.ndarray, None, None]:
    """Recursively interpolates between two frames."""
    if num_recursions == 0:
        yield frame1
    else:
        time = np.full((1,), 0.5, dtype=np.float32)
        mid_frame = interpolator(
            frame1[np.newaxis, ...],
            frame2[np.newaxis, ...],
            time
        )[0]
        progress["count"] += 1
        # Update progress using tqdm's write (prints without interfering with the bar).
        tqdm.write(f"GPU {gpu_info}: Processed {progress['count']}/{total} interpolation steps")
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                        interpolator, total, progress, gpu_info)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                        interpolator, total, progress, gpu_info)

def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator
) -> Iterable[np.ndarray]:
    """Generates interpolated frames from file paths."""
    n = len(frames)
    total = (n - 1) * (2**times_to_interpolate - 1)
    device = get_valid_device()
    logger.info("Starting recursive interpolation on %s with total steps: %d", device, total)
    progress = {"count": 0}
    for i in range(1, n):
        yield from _recursive_generator(
            read_image(frames[i - 1]),
            read_image(frames[i]),
            times_to_interpolate,
            interpolator,
            total,
            progress,
            device
        )
    yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator
) -> Iterable[np.ndarray]:
    """Generates interpolated frames from in-memory images."""
    n = len(frames)
    total = (n - 1) * (2**times_to_interpolate - 1)
    device = get_valid_device()
    logger.info("Starting recursive interpolation on %s with total steps: %d", device, total)
    progress = {"count": 0}
    for i in range(1, n):
        yield from _recursive_generator(
            frames[i - 1],
            frames[i],
            times_to_interpolate,
            interpolator,
            total,
            progress,
            device
        )
    yield frames[-1]

def get_ffmpeg_path() -> str:
    path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
    if not path:
        raise RuntimeError(
            f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' not found; please install ffmpeg."
        )
    return path
