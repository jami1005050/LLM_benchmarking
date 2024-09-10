import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from codecarbon import EmissionsTracker
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"
def reformat_func(example):
    question = example['text']
    image_input = example['image']
    if len(image_input.size) == 2:
        image_input = image_input.convert("RGB")
    return {'prompt': question,'image':image_input}


# Custom dataset class
class DiffusionDBDataset():
    def __init__(self):
        pass

    def __call__(self,data):
        text_batch = [element["prompt"] for element in data]
        image_batch = [element["image"] for element in data]
        return text_batch,image_batch
    

def main(exp_name, batch_size, model_name, base_dir = "output"):
    exp_id = exp_name + "_{}_batch_{}".format(model_name[10:], batch_size)
    print(exp_id)
    dataset = load_dataset("corto-ai/handwritten-text")
    column_names = dataset['train'].column_names

    dataset_selected = dataset["train"].select(np.arange(50))

    HF_token = ""
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    # load image from the IAM database
    processor = TrOCRProcessor.from_pretrained(model_name,token = HF_token) #used default one 
    model = VisionEncoderDecoderModel.from_pretrained(model_name,token = HF_token) # used default one as it is in huggingface 

    # training

    model = model.to("cuda")

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
    output_list = []
    input_list_pixel = []
    input_list_decode = []
    for index, (prompt,image) in enumerate(tqdm(train_dataloader)):
        # Generate images using Stable Diffusion Pipeline
        pixel_values = processor(image, return_tensors="pt").pixel_values  # Batch size 1
        pixel_values = pixel_values.to("cuda")
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
        decoder_input_ids = decoder_input_ids.to("cuda")
        with EmissionsTracker(project_name = exp_id, output_file = base_dir+"/emissions_L4{}.csv".format(exp_id), log_level="error", gpu_ids=cuda_device) as tracker:
            outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        output_list  += [outputs]

        input_list_decode += [decoder_input_ids.to("cpu").detach()]
        input_list_pixel += [pixel_values.to("cpu").detach()]
    restul_dict = {"Pixel": input_list_pixel,"DecodedInId":input_list_decode, "Output": output_list}
    torch.save(restul_dict, base_dir+"/inference_result_L4{}.pth".format(exp_id))


if __name__ == "__main__":
    exp_name   = "exGenImage"
    model_name = "microsoft/trocr-small-stage1"
    batch_size = 1
    base_dir   = "output/exGenImage"
    os.makedirs(base_dir, exist_ok=True)

    main(exp_name, batch_size, model_name, base_dir=base_dir)
