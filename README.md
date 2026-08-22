# LLM Benchmarking: Energy & Carbon Profiling of Generative AI Inference

Benchmarking scripts that measure the **latency, GPU power draw, and energy consumption** of six popular generative AI models — a text LLM, four diffusion models, and an OCR transformer — across two cloud GPU tiers (NVIDIA L4 and A100). This pipeline produced the cloud-side measurements used in a published sustainability study on generative AI inference.

## Published research

This benchmarking methodology is the cloud-measurement pipeline behind:

> Pengfei Li, **Mohammad J. Islam**, and Shaolei Ren. **"A Case Study of Environmental Footprints for Generative AI Inference: Cloud versus Edge."** *ACM SIGMETRICS Performance Evaluation Review*, 2025. [doi.org/10.1145/3764944.3764950](https://dl.acm.org/doi/10.1145/3764944.3764950)

**Study goal:** quantify whether running generative AI inference on edge devices (smartphones) is more environmentally sustainable than running it in the cloud (data-center GPUs), by measuring energy consumption and estimating the resulting carbon and water footprint for identical models and inputs on both platforms.

**Headline findings of the study:** shifting inference from a cloud A100 GPU to an on-device NPU (Samsung Galaxy S24) cut energy use by **over 90%**, and reduced the modeled carbon and water footprint by **more than 80%**, across all six models tested.

The full cloud-vs-edge comparison (including on-device NPU profiling and the carbon/water footprint model) lives in the companion repository, [lipengfeizju/LLM_energy_measure](https://github.com/lipengfeizju/LLM_energy_measure), which I also contributed to. **This repository is the standalone cloud-side half of that pipeline** — the scripts here generate the NVIDIA L4 / A100 latency and energy numbers reported in the paper's Table 1.

## What this repository does

For each of six models, the scripts here:
1. Load 50 fixed prompts (or images) from a task-appropriate Hugging Face dataset.
2. Run inference on the sampled inputs, one at a time (no batching, for a fair per-query comparison).
3. Wrap each inference call in a [CodeCarbon](https://github.com/mlco2/codecarbon) `EmissionsTracker` context manager to record wall-clock duration, GPU power draw, and energy consumed (Joules).
4. Save per-model results (model outputs + an emissions CSV) for later analysis in the visualization notebooks.

The same scripts were run unmodified on two Google Colab GPU configurations (L4 and A100) to produce a like-for-like hardware comparison.

## Models & datasets

| Task | Model | Hugging Face checkpoint | Dataset used |
|---|---|---|---|
| Text generation | Llama-2-7B-chat | `meta-llama/Llama-2-7b-chat-hf` | `vicgalle/alpaca-gpt4` (instruction prompts) |
| Text-to-image | Stable Diffusion v1.5 | `runwayml/stable-diffusion-v1-5` | `poloclub/diffusiondb` (image prompts) |
| Text-to-image | Stable Diffusion v2.1 | `stabilityai/stable-diffusion-2-1` | `poloclub/diffusiondb` |
| Text-to-audio | Riffusion | `riffusion/riffusion-model-v1` | `poloclub/diffusiondb` |
| Conditioned image generation | ControlNet (Canny edges) | `lllyasviel/sd-controlnet-canny` | `poloclub/diffusiondb` (+ Canny edge preprocessing) |
| Optical character recognition | TrOCR | `microsoft/trocr-small-stage1` | `corto-ai/handwritten-text` |

## Key findings

The table below reproduces the cloud-hardware comparison from the published study (Table 1), generated using this repository's scripts on Google Colab. Energy is per-prompt, in Joules.

| Model | L4 latency (s) | L4 energy (J) | A100 latency (s) | A100 energy (J) |
|---|---|---|---|---|
| TrOCR | 0.03 | 3.24 | 0.02 | 3.39 |
| Stable Diffusion v1.5 | 3.35 | 539.36 | 2.24 | 714.86 |
| Stable Diffusion v2.1 | 6.95 | 1118.12 | 2.27 | 1154.89 |
| Riffusion | 3.38 | 542.60 | 2.27 | 719.95 |
| ControlNet | 6.64 | 1067.77 | 3.52 | 1292.34 |
| Llama-2-7B | 23.56 | 3477.18 | 22.00 | 3803.71 |

**Takeaway:** the A100 is consistently faster (lower latency) than the L4 across every model, but that speed comes at a higher energy cost per query — the more power-hungry A100 draws enough more power that its per-query energy consumption is higher despite the shorter runtime. This latency-vs-energy trade-off between GPU tiers was the motivating observation for extending the comparison to edge devices in the full study, where on-device inference cut energy use by over 90% relative to the A100.

## Repository structure

| File | Description |
|---|---|
| `Llama_2_7B.py` | Profiles Llama-2-7B-chat text generation |
| `Stable_diffusion_V15.py` | Profiles Stable Diffusion v1.5 image generation |
| `Stable_diffusion_V21.py` | Profiles Stable Diffusion v2.1 image generation |
| `Riffusion.py` | Profiles Riffusion text-to-spectrogram generation |
| `ControlNet.py` | Profiles Canny-edge-conditioned ControlNet image generation |
| `TrOCR.py` | Profiles TrOCR handwritten-text recognition |
| `visualization.ipynb` | Plots GPU power, duration, energy, and CDFs for the image/OCR models |
| `visualization_llm.ipynb` | Plots energy, throughput (tokens/s), and energy-per-token for Llama-2-7B |

## Setup

Requires Python 3.8+ and a CUDA-capable GPU (developed and profiled on Google Colab with NVIDIA L4 and A100 instances).

```bash
pip install codecarbon
pip install torch torchvision torchaudio --upgrade
pip install -U bitsandbytes==0.42.0
pip install -U peft==0.8.2
pip install -U trl==0.7.10
pip install -U accelerate==0.27.1
pip install -U datasets==2.17.0
pip install -U transformers==4.38.1
pip install diffusers
```

Some models (e.g. Llama-2-7B) are gated on Hugging Face — set your access token in the `user_token` / `HF_token` variable near the top of the relevant script before running.

## Usage

Each script is self-contained and creates its own output directory:

```bash
python Llama_2_7B.py
python Stable_diffusion_V15.py
python Stable_diffusion_V21.py
python Riffusion.py
python ControlNet.py
python TrOCR.py
```

Each run writes an `emissions_*.csv` (per-prompt latency/power/energy from CodeCarbon) and an `inference_result_*.pth` (model inputs/outputs) to an `output/` subdirectory. Load these into `visualization.ipynb` / `visualization_llm.ipynb` to reproduce the plots.

## My role

I designed and implemented this end-to-end benchmarking pipeline — dataset loading, per-model inference harnesses, and CodeCarbon instrumentation for all six models — as part of a sustainability research project in [Prof. Shaolei Ren's lab](https://shaoleiren.github.io/) at UC Riverside. The resulting cloud GPU measurements are reported in Table 1 of our published paper (co-authored with Pengfei Li and Prof. Shaolei Ren), which also incorporates on-device edge measurements collected in the companion [LLM_energy_measure](https://github.com/lipengfeizju/LLM_energy_measure) repository.
