## Picked Models
### Text Generation
1. Llama2-7B
### Image Generation 
1. ControlNet--lllyasviel/sd-controlnet-canny
2. Stable-Diffusion-v2.1--stabilityai/stable-diffusion-2-1
### MultiModal
1. OpenAI-Clip
2. TrOCR

### Data set 
1. poloclub/diffusiondb -->Text to image
2. corto-ai/handwritten-text  --> Image to text
3. vicgalle/alpaca-gpt4    --> Text generation


### Packages 
!pip install codecarbon 
!pip3 install torch torchvision torchaudio --upgrade
!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.8.2
!pip3 install -q -U trl==0.7.10
!pip3 install -q -U accelerate==0.27.1
!pip3 install -q -U datasets==2.17.0
!pip3 install -q -U transformers==4.38.1
!pip install diffusers

# Inference Setup:
For each model, 50 input prompts are sampled to evaluate inference performance. We capture key metrics for each prompt, including:

 Latency: Time taken for inference completion per prompt.< br / > 
 Energy Consumption: Energy used during the inference process.< br / > 
 Carbon Emission: Carbon footprint associated with the energy consumed.

### These metrics are collected across two distinct hardware configurations hosted on the Google Cloud Platform:

L4 GPU< br / > 
A100 GPU< br / > 
The comparison of these two GPUs provides insights into the trade-offs in terms of speed (latency), energy efficiency, and environmental impact (carbon emissions) when deploying machine learning models on different hardware infrastructures.
