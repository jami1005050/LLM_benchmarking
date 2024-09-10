import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from codecarbon import EmissionsTracker
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import pandas as pd


os.environ["TOKENIZERS_PARALLELISM"] = "false"
def reformat_func(example):

    question = example['prompt']

    return {'prompt': question}


# Custom dataset class
class DiffusionDBDataset():
    def __init__(self):
        pass

    def __call__(self,data):
        text_batch = [element["prompt"] for element in data]

        return text_batch



def main(exp_name, batch_size, model_name, base_dir = "output"):
    exp_id = exp_name + "_{}_batch_{}".format(model_name[12:], batch_size)

    model_path = "pretrained"
    dataset_path = "dataset"
    dataset = load_dataset("poloclub/diffusiondb", cache_dir=dataset_path)
    column_names = dataset['train'].column_names

    dataset_selected = dataset["train"].select(np.arange(5))


    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = DiffusionPipeline.from_pretrained(model_name,
                                            torch_dtype=torch.float16,
                                            cache_dir=model_path)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    processed_dataset = dataset_selected.map(
        lambda example: reformat_func(example),
        batched=False,
        remove_columns=column_names
    )
    train_dataset = DiffusionDBDataset()
    # Create a dataloader

    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
    print("CPU allocated : ", cuda_device)


    train_dataloader = DataLoader(
        dataset=processed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_dataset
    )
    count = 0
    prompt_list = []
    for index, (prompts) in enumerate(tqdm(train_dataloader)):
        # Generate images using Stable Diffusion Pipeline
        prompt_list+= [prompts]
        with EmissionsTracker(project_name = exp_id, output_file = base_dir+"/emissions_L4{}.csv".format(exp_id), log_level="error", gpu_ids=cuda_device) as tracker:
            generated_images = pipe(prompts).images
        # generated_images.to("cpu").detach()
        generated_images[0].save("riffusion_image_"+str(count)+".png")
        count+=1

    prompt_list = pd.DataFrame({
          'Prompt': prompt_list,
      })
    prompt_list.to_csv("riffusion_prompt_list.csv")



if __name__ == "__main__":

    exp_name   = "exGenImage"
    model_name = "riffusion/riffusion-model-v1"
    batch_size = 1
    base_dir   = "output/exGenImage"
    os.makedirs(base_dir, exist_ok=True)

    main(exp_name, batch_size, model_name, base_dir=base_dir)