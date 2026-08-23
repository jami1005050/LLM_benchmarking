# LLM Benchmarking

A benchmarking toolkit for measuring the energy cost of generative AI inference.

Every generated image, spectrogram, or block of text costs real electricity to produce, but that cost is almost never visible — a model card tells you accuracy and speed, not joules per query. This repo is a small, self-contained toolkit for closing that gap: it runs six popular generative models — a text LLM, four image/audio diffusion models, and an OCR transformer — through an identical measurement pipeline, and records exactly how much latency, GPU power, and energy each one burns per prompt, on two different GPU tiers.

## What this measures

Wall-clock time alone hides a lot: a faster GPU can still be the more expensive one to run, once you account for how much power it draws while it works. So instead of just timing inference, every run here is wrapped in [CodeCarbon](https://github.com/mlco2/codecarbon)'s `EmissionsTracker`, which samples GPU/CPU/RAM power draw for the duration of each call and turns it into an energy figure per prompt. Comparing that number across an NVIDIA L4 and an A100, for the same model and the same inputs, is what actually tells you whether "faster" and "greener" point the same direction.

The six models, and the datasets used to generate their inputs:

| Task | Model | Checkpoint | Dataset |
|---|---|---|---|
| Text generation | Llama-2-7B-chat | `meta-llama/Llama-2-7b-chat-hf` | `vicgalle/alpaca-gpt4` |
| Text-to-image | Stable Diffusion v1.5 | `runwayml/stable-diffusion-v1-5` | `poloclub/diffusiondb` |
| Text-to-image | Stable Diffusion v2.1 | `stabilityai/stable-diffusion-2-1` | `poloclub/diffusiondb` |
| Text-to-audio | Riffusion | `riffusion/riffusion-model-v1` | `poloclub/diffusiondb` |
| Edge-conditioned image generation | ControlNet (Canny) | `lllyasviel/sd-controlnet-canny` | `poloclub/diffusiondb` |
| Optical character recognition | TrOCR | `microsoft/trocr-small-stage1` | `corto-ai/handwritten-text` |

Each model is profiled on 50 sampled prompts, one at a time (no batching), so the comparison is apples-to-apples across models and hardware.

## How it's built

Six models with six different pipelines still share the same three moving parts: sample N examples from a dataset, run them through a model one at a time, and record what that cost. Rather than duplicate that scaffolding six times, it lives in one place, in `utils/`:

- `utils/config.py` — a small registry of what each model is: its checkpoint, its dataset, and the exact string used to name its output files.
- `utils/data.py` — dataset sampling and the batch-collation logic shared across the prompt-only and prompt-plus-image models.
- `utils/tracking.py` — wraps `EmissionsTracker` and owns the output-path convention, so the GPU tag and output directory are real parameters instead of values hardcoded per script.
- `utils/cli.py` — the command-line flags every script shares (`--device-tag`, `--num-samples`, `--batch-size`, `--exp-name`, `--base-dir`, `--hf-token`).

Each of the four entry-point scripts (`llama_bench.py`, `diffusion_bench.py`, `controlnet_bench.py`, `trocr_bench.py`) imports from `utils/` and adds only what's specific to its own model — dataset preprocessing, the actual model call. `diffusion_bench.py` in particular replaces three earlier scripts that were nearly identical except for a checkpoint name; it now covers Stable Diffusion v1.5, v2.1, and Riffusion behind a single `--model` flag.

## Running a benchmark

```bash
pip install -r requirements.txt
```

Llama-2-7B is a gated checkpoint — export an access token first:

```bash
export HF_TOKEN=your_huggingface_token
```

Then run any of the four benchmarks, tagging each run with the hardware it's on:

```bash
python llama_bench.py       --device-tag A100
python diffusion_bench.py   --device-tag A100 --model sd15   # or sd21, riffusion
python controlnet_bench.py  --device-tag A100
python trocr_bench.py       --device-tag A100
```

`--num-samples` (default 50) and `--batch-size` (default 1) are also overridable. Each run creates `output/<exp-name>/`, containing the CodeCarbon emissions CSV, any generated images, and — for Llama and TrOCR — a `.pth` file with the raw model inputs/outputs.

## Visualizing results

`visualization.ipynb` and `visualization_llm.ipynb` load the CSVs from `output/` and plot GPU power, duration, energy, and energy-per-token across models and devices, along with CDFs of per-prompt energy use. The consistent pattern across runs: the more powerful GPU tier is reliably faster, but not reliably cheaper — the extra power it draws often outweighs the time it saves, so the "best" hardware choice depends on whether you're optimizing for latency or for energy.

## Repository structure

| Path | Contents |
|---|---|
| `llama_bench.py` | Llama-2-7B-chat text generation |
| `diffusion_bench.py` | Stable Diffusion v1.5 / v2.1 / Riffusion (`--model`) |
| `controlnet_bench.py` | Canny-edge-conditioned ControlNet |
| `trocr_bench.py` | TrOCR handwritten-text recognition |
| `utils/` | Shared config, dataset sampling, tracking, and CLI helpers |
| `visualization.ipynb` | Plots for the diffusion/ControlNet/TrOCR runs |
| `visualization_llm.ipynb` | Plots for the Llama-2-7B runs |

## License

MIT — see [LICENSE](LICENSE).

---

This benchmarking approach also fed into a peer-reviewed study on the environmental footprint of cloud vs. edge generative AI inference ([ACM SIGMETRICS PER, 2025](https://dl.acm.org/doi/10.1145/3764944.3764950)), which extends the same methodology to on-device measurement.
