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

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

# Set up a logger for this module.
logger = logging.getLogger("frame_interpolation.util")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def read_image(filename: str) -> np.ndarray:
    """Reads an 8-bit sRGB image and returns a float32 array in [0,1]."""
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image_np = tf.cast(image, dtype=tf.float32).numpy()
    return image_np / _UINT8_MAX_F

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 image (values in [0,1]) to a file as PNG or JPEG."""
    image_uint8 = (np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F) + 0.5).astype(np.uint8)
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.jpg':
        image_data = tf.io.encode_jpeg(image_uint8)
    else:
        image_data = tf.io.encode_png(image_uint8)
    tf.io.write_file(filename, image_data)

def output_frames(frames: List[np.ndarray], frames_dir: str):
    """Writes a list of frames to PNG files in the given directory using a tqdm progress bar."""
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logger.info('Removing existing frames from %s.', frames_dir)
            for f in old_frames:
                tf.io.gfile.remove(f)
    else:
        tf.io.gfile.makedirs(frames_dir)
    for idx, frame in tqdm(enumerate(frames), total=len(frames), ncols=100, desc="Saving frames", colour='blue'):
        write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logger.info('Output frames saved in %s.', frames_dir)

def _recursive_generator(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    interpolator: interpolator_lib.Interpolator
) -> Generator[np.ndarray, None, None]:
    """Recursively interpolates between two frames."""
    if num_recursions == 0:
        yield frame1
    else:
        t_val = np.full((1,), 0.5, dtype=np.float32)
        mid_frame = interpolator(frame1[np.newaxis, ...], frame2[np.newaxis, ...], t_val)[0]
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator)

def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator,
    gpu_info: Optional[str] = None
) -> Iterable[np.ndarray]:
    """
    Reads image files and recursively interpolates between each consecutive pair.
    Yields the sequence of frames.
    """
    n = len(frames)
    if gpu_info is None:
        gpu_info = tf.test.gpu_device_name() or "CPU"
    logger.info("Starting recursive interpolation on %s.", gpu_info)
    for i in range(1, n):
        yield from _recursive_generator(read_image(frames[i - 1]), read_image(frames[i]), times_to_interpolate, interpolator)
    yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator,
    gpu_info: Optional[str] = None
) -> Iterable[np.ndarray]:
    """
    Recursively interpolates between in-memory frames.
    Yields the sequence of frames.
    """
    n = len(frames)
    if gpu_info is None:
        gpu_info = tf.test.gpu_device_name() or "CPU"
    logger.info("Starting recursive interpolation on %s.", gpu_info)
    for i in range(1, n):
        yield from _recursive_generator(frames[i - 1], frames[i], times_to_interpolate, interpolator)
    yield frames[-1]
