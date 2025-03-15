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
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'


def read_image(filename: str) -> np.ndarray:
  """Reads an sRgb 8-bit image.

  Args:
    filename: The input filename to read.

  Returns:
    A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_data = tf.io.read_file(filename)
  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


def write_image(filename: str, image: np.ndarray) -> None:
  """Writes a float32 3-channel RGB ndarray image, with colors in range [0..1].

  Args:
    filename: The output filename to save.
    image: A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
  """
  image_in_uint8_range = np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
  image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)

  extension = os.path.splitext(filename)[1]
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
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.
    total: Total expected progress count.
    progress: A dict containing a mutable progress counter, e.g. {"count": 0}.
    gpu_info: A string indicating GPU info (e.g. '/device:GPU:0').

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Prepare time input for midpoint interpolation.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        frame1[np.newaxis, ...],
        frame2[np.newaxis, ...],
        time
    )[0]
    # Update progress counter.
    progress["count"] += 1
    # Log progress every 10 steps or when finished.
    if progress["count"] % 10 == 0 or progress["count"] == total:
      logger.info("GPU %s: Processed %d/%d interpolation steps", gpu_info, progress["count"], total)
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator, total, progress, gpu_info)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator, total, progress, gpu_info)


def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator
) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Loads the files on demand and uses the yield paradigm to return the frames
  to allow streamed processing of longer videos.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range [0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  total = (n - 1) * (2**times_to_interpolate - 1)
  # Obtain GPU device name if available.
  gpu_info = tf.test.gpu_device_name() or "CPU"
  logger.info("Starting recursive interpolation on GPU: %s with total steps: %d", gpu_info, total)
  # Create a mutable progress counter.
  progress = {"count": 0}
  for i in range(1, n):
    yield from _recursive_generator(
        read_image(frames[i - 1]),
        read_image(frames[i]),
        times_to_interpolate,
        interpolator,
        total,
        progress,
        gpu_info
    )
  # Separately yield the final frame.
  yield read_image(frames[-1])


def interpolate_recursively_from_memory(
    frames: List[np.ndarray],
    times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator
) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  This is functionally equivalent to interpolate_recursively_from_files(), but
  expects the input frames in memory, instead of loading them on demand.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range [0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  total = (n - 1) * (2**times_to_interpolate - 1)
  gpu_info = tf.test.gpu_device_name() or "CPU"
  logger.info("Starting recursive interpolation on GPU: %s with total steps: %d", gpu_info, total)
  progress = {"count": 0}
  for i in range(1, n):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    times_to_interpolate, interpolator,
                                    total, progress, gpu_info)
  yield frames[-1]


def get_ffmpeg_path() -> str:
  path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
  if not path:
    raise RuntimeError(
        f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' is not found; "
        "perhaps install ffmpeg using 'apt-get install ffmpeg'.")
  return path
