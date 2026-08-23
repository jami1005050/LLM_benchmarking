"""Benchmark harness for text-to-image / text-to-audio diffusion models.

Profiles one of Stable Diffusion v1.5, Stable Diffusion v2.1, or Riffusion
(selected with --model) on a sample of DiffusionDB prompts, recording
latency and energy consumption per prompt with CodeCarbon.
"""

import argparse
import os

import pandas as pd
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.cli import add_common_bench_args, resolve_base_dir
from utils.config import DIFFUSION_MODELS
from utils.data import PromptOnlyCollate, load_sampled_dataset
from utils.tracking import artifact_path, build_exp_id, make_tracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reformat_func(example):
    return {"prompt": example["prompt"]}


def main(model_key, device_tag, num_samples, batch_size, exp_name, base_dir, hf_token):
    model_cfg = DIFFUSION_MODELS[model_key]
    exp_id = build_exp_id(exp_name, model_cfg.short_name, batch_size)

    dataset = load_sampled_dataset(model_cfg.dataset, num_samples, reformat_func)

    pipe = DiffusionPipeline.from_pretrained(
        model_cfg.checkpoint,
        torch_dtype=torch.float16,
        cache_dir="pretrained",
        token=hf_token or None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PromptOnlyCollate(),
    )

    prompt_log = []
    count = 0
    for prompts in tqdm(dataloader):
        prompt_log += prompts
        with make_tracker(exp_id, base_dir, device_tag):
            generated_images = pipe(prompts).images
        for image in generated_images:
            image.save(artifact_path(base_dir, f"{model_cfg.key}_image_{count}.png", subdir="images"))
            count += 1

    prompt_csv = artifact_path(base_dir, f"{model_cfg.key}_prompt_list.csv")
    pd.DataFrame({"Prompt": prompt_log}).to_csv(prompt_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(DIFFUSION_MODELS),
                         help="Which diffusion model to profile.")
    add_common_bench_args(parser, default_exp_name="exGenImage")
    args = parser.parse_args()

    main(
        model_key=args.model,
        device_tag=args.device_tag,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        exp_name=args.exp_name,
        base_dir=resolve_base_dir(args),
        hf_token=args.hf_token,
    )
