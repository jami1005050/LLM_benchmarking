"""Benchmark harness for ControlNet (Canny-edge-conditioned image generation).

Profiles lllyasviel/sd-controlnet-canny, built on the Stable Diffusion v1.5
base pipeline, on a sample of DiffusionDB prompts and their Canny-edge maps,
recording latency and energy consumption per prompt with CodeCarbon.
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.cli import add_common_bench_args, resolve_base_dir
from utils.config import CONTROLNET, DIFFUSION_MODELS
from utils.data import PromptImageCollate, load_sampled_dataset
from utils.tracking import artifact_path, build_exp_id, make_tracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200


def reformat_func(example):
    original_image = load_image(example["image"])
    edges = cv2.Canny(np.array(original_image), CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edges = np.concatenate([edges[:, :, None]] * 3, axis=2)
    return {"prompt": example["prompt"], "image": Image.fromarray(edges)}


def main(device_tag, num_samples, batch_size, exp_name, base_dir, hf_token):
    exp_id = build_exp_id(exp_name, CONTROLNET.short_name, batch_size)

    dataset = load_sampled_dataset(CONTROLNET.dataset, num_samples, reformat_func)

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET.checkpoint,
        torch_dtype=torch.float16,
        token=hf_token or None,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        DIFFUSION_MODELS["sd15"].checkpoint,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir="pretrained",
        token=hf_token or None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PromptImageCollate(),
    )

    prompt_log = []
    count = 0
    for prompts, images in tqdm(dataloader):
        prompt_log += prompts
        with make_tracker(exp_id, base_dir, device_tag):
            generated_images = pipe(prompts, image=images).images
        for image in generated_images:
            image.save(artifact_path(base_dir, f"{CONTROLNET.key}_image_{count}.png", subdir="images"))
            count += 1

    prompt_csv = artifact_path(base_dir, f"{CONTROLNET.key}_prompt_list.csv")
    pd.DataFrame({"Prompt": prompt_log}).to_csv(prompt_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_bench_args(parser, default_exp_name=CONTROLNET.default_exp_name)
    args = parser.parse_args()

    main(
        device_tag=args.device_tag,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        exp_name=args.exp_name,
        base_dir=resolve_base_dir(args),
        hf_token=args.hf_token,
    )
