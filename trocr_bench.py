"""Benchmark harness for TrOCR handwritten-text recognition.

Profiles microsoft/trocr-small-stage1 on a sample of handwritten-text
images, recording latency and energy consumption per prompt with
CodeCarbon.
"""

import argparse
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.cli import add_common_bench_args, resolve_base_dir
from utils.config import TROCR
from utils.data import PromptImageCollate, load_sampled_dataset
from utils.tracking import build_exp_id, inference_result_path, make_tracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reformat_func(example):
    image = example["image"]
    if len(image.size) == 2:
        image = image.convert("RGB")
    return {"prompt": example["text"], "image": image}


def main(device_tag, num_samples, batch_size, exp_name, base_dir, hf_token):
    exp_id = build_exp_id(exp_name, TROCR.short_name, batch_size)

    dataset = load_sampled_dataset(TROCR.dataset, num_samples, reformat_func)

    processor = TrOCRProcessor.from_pretrained(TROCR.checkpoint, token=hf_token or None)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR.checkpoint, token=hf_token or None)
    model = model.to("cuda")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PromptImageCollate(),
    )

    outputs, pixel_inputs, decoder_inputs = [], [], []
    for _, images in tqdm(dataloader):
        pixel_values = processor(images, return_tensors="pt").pixel_values.to("cuda")
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to("cuda")

        with make_tracker(exp_id, base_dir, device_tag):
            output = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

        outputs.append(output)
        pixel_inputs.append(pixel_values.to("cpu").detach())
        decoder_inputs.append(decoder_input_ids.to("cpu").detach())

    result = {"Pixel": pixel_inputs, "DecodedInId": decoder_inputs, "Output": outputs}
    torch.save(result, inference_result_path(base_dir, device_tag, exp_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_bench_args(parser, default_exp_name=TROCR.default_exp_name)
    args = parser.parse_args()

    main(
        device_tag=args.device_tag,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        exp_name=args.exp_name,
        base_dir=resolve_base_dir(args),
        hf_token=args.hf_token,
    )
