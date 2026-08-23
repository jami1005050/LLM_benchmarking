"""Shared dataset sampling and batch-collation helpers.

Every benchmark script samples a fixed number of examples from a Hugging
Face dataset and reformats them with a model-specific function before
batching. This module owns the part that's identical across scripts;
the model-specific `reformat_fn` stays in each script.
"""

import numpy as np
from datasets import load_dataset


def load_sampled_dataset(dataset_id, num_samples, reformat_fn, cache_dir="dataset", split="train"):
    """Load `dataset_id`, take the first `num_samples` rows, and reformat each
    with `reformat_fn`. Mirrors the load -> select -> map pattern every
    benchmark script used to duplicate, with `num_samples` as a real argument
    instead of a hardcoded `np.arange(...)` literal.
    """
    dataset = load_dataset(dataset_id, cache_dir=cache_dir)
    column_names = dataset[split].column_names
    selected = dataset[split].select(np.arange(num_samples))
    return selected.map(
        lambda example: reformat_fn(example),
        batched=False,
        remove_columns=column_names,
    )


class PromptOnlyCollate:
    """Collate function for datasets reformatted to a single 'prompt' field."""

    def __call__(self, data):
        return [element["prompt"] for element in data]


class PromptImageCollate:
    """Collate function for datasets reformatted to 'prompt' and 'image' fields."""

    def __call__(self, data):
        text_batch = [element["prompt"] for element in data]
        image_batch = [element["image"] for element in data]
        return text_batch, image_batch
