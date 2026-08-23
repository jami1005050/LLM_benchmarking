"""Benchmark harness for Llama-2-7B-chat text generation.

Profiles meta-llama/Llama-2-7b-chat-hf on a sample of Alpaca-GPT4
instructions, recording latency and energy consumption per prompt with
CodeCarbon.
"""

import argparse
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.cli import add_common_bench_args, resolve_base_dir
from utils.config import LLAMA
from utils.data import load_sampled_dataset
from utils.tracking import build_exp_id, inference_result_path, make_tracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_GENERATION_LENGTH = 1024


def reformat_func(example, prompt_prefix=" "):
    question = prompt_prefix + example["instruction"]
    if example.get("input", None) and example["input"].strip():
        question += f"\n{example['input']}"
    return {"question": question}


class DataCollatorForLLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        text_batch = [element["question"] for element in data]
        return self.tokenizer(text_batch, padding="longest", truncation=False, return_tensors="pt")


def main(device_tag, num_samples, batch_size, exp_name, base_dir, hf_token):
    exp_id = build_exp_id(exp_name, LLAMA.short_name, batch_size)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA.checkpoint, token=hf_token or None, cache_dir="pretrained")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA.checkpoint,
        token=hf_token or None,
        cache_dir="pretrained",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    dataset = load_sampled_dataset(LLAMA.dataset, num_samples, reformat_func)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLLM(tokenizer),
    )

    inputs, outputs = [], []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        with make_tracker(exp_id, base_dir, device_tag):
            generated = model.generate(input_ids, max_length=MAX_GENERATION_LENGTH)
        inputs.append(input_ids.to("cpu").detach())
        outputs.append(generated.to("cpu").detach())

    result = {"Input": inputs, "Output": outputs}
    torch.save(result, inference_result_path(base_dir, device_tag, exp_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_bench_args(parser, default_exp_name=LLAMA.default_exp_name)
    args = parser.parse_args()

    main(
        device_tag=args.device_tag,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        exp_name=args.exp_name,
        base_dir=resolve_base_dir(args),
        hf_token=args.hf_token,
    )
