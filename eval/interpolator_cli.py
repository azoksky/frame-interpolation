import os
import math
import numpy as np
import mediapy
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from eval import interpolator as interpolator_lib, util

def process_frames(frame_paths: list[str], times_to_interpolate: int, 
                   model_path: str, output_fps: int = 30, 
                   align: int = 64, block_height: int = 1, block_width: int = 1) -> str:
    """
    Interpolate between frames in frame_paths using multiple GPUs.
    Returns the filepath of the output video.
    """
    # Ensure we have at least two frames
    if len(frame_paths) < 2:
        raise ValueError("Need at least two frames to interpolate.")
    # Sort frame paths to ensure correct order
    frame_paths = sorted(frame_paths)

    # Enable memory growth on all GPUs to manage memory usage
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    num_gpus = len(physical_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU available for multi-GPU interpolation.")

    # If only one GPU or only one gap, fall back to single-GPU processing
    total_pairs = len(frame_paths) - 1  # number of adjacent frame pairs
    if num_gpus == 1 or total_pairs == 1:
        # Use single GPU interpolation (no parallelism needed or possible)
        interpolator = interpolator_lib.Interpolator(model_path, align, [block_height, block_width])
        frames = list(util.interpolate_recursively_from_files(frame_paths, times_to_interpolate, interpolator))
        output_path = "interpolated.mp4"
        mediapy.write_video(output_path, frames, fps=output_fps)
        return output_path

    # Determine how to split frame pairs among GPUs
    num_gpus_to_use = min(num_gpus, total_pairs)
    # Calculate approximately equal segment sizes (in terms of number of pairs per GPU)
    pairs_per_gpu = math.ceil(total_pairs / num_gpus_to_use)

    segments = []  # each segment is a list of frame file paths
    for i in range(num_gpus_to_use):
        start_index = i * pairs_per_gpu
        end_index = min(total_pairs, (i + 1) * pairs_per_gpu)
        if i < num_gpus_to_use - 1:
            # For all but the last segment, include one extra frame for overlap
            segment_frames = frame_paths[start_index : end_index + 1] 
        else:
            # Last segment goes till the final frame
            segment_frames = frame_paths[start_index : end_index + 1]
        segments.append(segment_frames)
    # Adjust segments to ensure overlap of one frame between consecutive segments
    # (The above logic already overlaps one frame except possibly between middle segments if any gap – ensure each next segment starts with the last frame of the previous)
    # We can explicitly enforce the overlap:
    for j in range(1, len(segments)):
        # Ensure the first frame of segment j is the same as last frame of segment j-1
        segments[j][0] = segments[j-1][-1]

    # Load one interpolator model per GPU
    interpolators = []
    for gpu_index in range(num_gpus_to_use):
        with tf.device(f'/GPU:{gpu_index}'):
            interpolators.append(interpolator_lib.Interpolator(model_path, align, [block_height, block_width]))
    # (Each Interpolator's model weights are loaded onto its specific GPU)

    # Define worker function for a thread to process its segment
    def process_segment(segment_index: int) -> list[np.ndarray]:
        gpu_idx = segment_index
        segment_frames = segments[segment_index]
        # Bind this thread's operations to the specific GPU
        with tf.device(f'/GPU:{gpu_idx}'):
            interp = interpolators[segment_index]
            # Run recursive interpolation on this segment of frames
            result_frames = list(util.interpolate_recursively_from_files(segment_frames, times_to_interpolate, interp))
        return result_frames

    # Run all segments in parallel threads
    results = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=len(segments)) as executor:
        futures = []
        for idx in range(len(segments)):
            futures.append(executor.submit(process_segment, idx))
        # Collect results
        for idx, fut in enumerate(futures):
            results[idx] = fut.result()

    # Merge results, dropping overlap frames to avoid duplicates
    final_frames = results[0]
    for seg_idx in range(1, len(results)):
        # Each segment output includes the first frame which is the last frame of previous segment
        # Skip that to avoid duplication
        overlap_frame = results[seg_idx][0]
        merged = results[seg_idx][1:]  # drop the first frame (overlap)
        final_frames.extend(merged)
    # `final_frames` now contains the full sequence in correct order

    # Write output video using mediapy
    output_path = "interpolated.mp4"
    mediapy.write_video(output_path, final_frames, fps=output_fps)
    return output_path

# Example usage (in a Gradio app, for instance):
# video_path = process_frames(list_of_frame_paths, times_to_interpolate=2, model_path="path/to/saved_model", output_fps=30)
# print(f"Output video saved at {video_path}")
