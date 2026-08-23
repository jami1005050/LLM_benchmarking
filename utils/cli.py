"""Shared command-line argument handling for the benchmark scripts."""

import os


def add_common_bench_args(parser, default_exp_name):
    parser.add_argument("--device-tag", required=True,
                         help="Label for the hardware this run is on, e.g. L4, A100 (embedded in output filenames).")
    parser.add_argument("--num-samples", type=int, default=50,
                         help="Number of prompts/images to sample from the dataset.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--exp-name", default=default_exp_name)
    parser.add_argument("--base-dir", default=None,
                         help="Output directory. Defaults to output/<exp-name>.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                         help="Hugging Face access token, for gated checkpoints. Defaults to $HF_TOKEN.")
    return parser


def resolve_base_dir(args):
    return args.base_dir or os.path.join("output", args.exp_name)
