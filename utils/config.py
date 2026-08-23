"""Registry of the models and datasets this harness benchmarks.

Every entry's `short_name` is an explicit literal (never derived by slicing
a checkpoint string), so filename construction in `tracking.py` can't drift
out of sync with the actual checkpoint being profiled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    key: str            # short identifier used in output filenames, e.g. "sd15"
    checkpoint: str      # Hugging Face checkpoint id
    short_name: str       # human-readable name embedded in emissions filenames
    dataset: str            # Hugging Face dataset id used to sample prompts/images
    default_exp_name: str    # default --exp-name for this script


DIFFUSION_MODELS = {
    "sd15": ModelConfig(
        key="sd15",
        checkpoint="runwayml/stable-diffusion-v1-5",
        short_name="stable-diffusion-v1-5",
        dataset="poloclub/diffusiondb",
        default_exp_name="exGenImage",
    ),
    "sd21": ModelConfig(
        key="sd21",
        checkpoint="stabilityai/stable-diffusion-2-1",
        short_name="stable-diffusion-2-1",
        dataset="poloclub/diffusiondb",
        default_exp_name="exGenImage",
    ),
    "riffusion": ModelConfig(
        key="riffusion",
        checkpoint="riffusion/riffusion-model-v1",
        short_name="riffusion-model-v1",
        dataset="poloclub/diffusiondb",
        default_exp_name="exGenImage",
    ),
}

CONTROLNET = ModelConfig(
    key="controlnet",
    checkpoint="lllyasviel/sd-controlnet-canny",
    short_name="sd-controlnet-canny",
    dataset="poloclub/diffusiondb",
    default_exp_name="exGenImage",
)

TROCR = ModelConfig(
    key="trocr",
    checkpoint="microsoft/trocr-small-stage1",
    short_name="trocr-small-stage1",
    dataset="corto-ai/handwritten-text",
    default_exp_name="exGenImage",
)

LLAMA = ModelConfig(
    key="llama2-7b",
    checkpoint="meta-llama/Llama-2-7b-chat-hf",
    short_name="Llama-2-7b-chat-hf",
    dataset="vicgalle/alpaca-gpt4",
    default_exp_name="exp5",
)
