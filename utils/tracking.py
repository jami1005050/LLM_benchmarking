"""Emissions tracking and output-path conventions shared by every script.

Centralizing the filename/path logic here means the GPU tag and sample
count are real parameters instead of literals hardcoded per script, and
every artifact (emissions CSV, inference result, generated images, prompt
lists) consistently lands under `base_dir` instead of the current working
directory.
"""

import os

from codecarbon import EmissionsTracker


def build_exp_id(exp_name, short_name, batch_size):
    return f"{exp_name}_{short_name}_batch_{batch_size}"


def emissions_csv_path(base_dir, device_tag, exp_id):
    return os.path.join(base_dir, f"emissions_{device_tag}{exp_id}.csv")


def inference_result_path(base_dir, device_tag, exp_id):
    return os.path.join(base_dir, f"inference_result_{device_tag}{exp_id}.pth")


def artifact_path(base_dir, filename, subdir=None):
    """Return a path for a non-tracked output (image, prompt-list CSV, ...),
    creating its parent directory under `base_dir` if needed.
    """
    directory = os.path.join(base_dir, subdir) if subdir else base_dir
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def make_tracker(exp_id, base_dir, device_tag, gpu_ids=None, log_level="error"):
    """Build an EmissionsTracker that writes to the shared path convention."""
    os.makedirs(base_dir, exist_ok=True)
    if gpu_ids is None:
        gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES")
    return EmissionsTracker(
        project_name=exp_id,
        output_file=emissions_csv_path(base_dir, device_tag, exp_id),
        log_level=log_level,
        gpu_ids=gpu_ids,
    )
