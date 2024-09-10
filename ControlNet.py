from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from codecarbon import EmissionsTracker
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def reformat_func(example):

    question = example['prompt']
    original_image = load_image(example['image'])

    image = np.array(original_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return {'prompt': question,'image':canny_image}


# Custom dataset class
class DiffusionDBDataset():
    def __init__(self):
        pass

    def __call__(self,data):
        text_batch = [element["prompt"] for element in data]
        image_batch = [element["image"] for element in data]
        return text_batch,image_batch
    


def main(exp_name, batch_size, model_name, base_dir = "output"):
    exp_id = exp_name + "_{}_batch_{}".format(model_name[11:], batch_size)
    print(exp_id)
    model_path = "pretrained"
    dataset_path = "dataset"
    dataset = load_dataset("poloclub/diffusiondb", cache_dir=dataset_path)
    column_names = dataset['train'].column_names
    HF_token = '' # YOUR TOKEN
    dataset_selected = dataset["train"].select(np.arange(50))


    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    controlnet = ControlNetModel.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                token = HF_token)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=torch.float16,token = HF_token,cache_dir=model_path
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

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
    image_list = []
    for index, (prompts,images) in enumerate(tqdm(train_dataloader)):
        # Generate images using Stable Diffusion Pipeline
        prompt_list+= [prompts]
        image_list+= [images]
        with EmissionsTracker(project_name = exp_id, output_file = base_dir+"/emissions_A100{}.csv".format(exp_id), log_level="error", gpu_ids=cuda_device) as tracker:
            generated_images = pipe(prompts,image=images).images
        # generated_images.to("cpu").detach()
        generated_images[0].save("control_netimage_"+str(count)+".png")
        count+=1

    prompt_list = pd.DataFrame({
          'Prompt': prompt_list,
          'Image': image_list
      })
    prompt_list.to_csv("control_net_prompt_list.csv")


if __name__ == "__main__":

    exp_name   = "exGenImage"
    model_name = "lllyasviel/sd-controlnet-canny"
    batch_size = 1
    base_dir   = "output/exGenImage"
    os.makedirs(base_dir, exist_ok=True)

    main(exp_name, batch_size, model_name, base_dir=base_dir)