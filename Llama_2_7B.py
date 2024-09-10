import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codecarbon import EmissionsTracker
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def reformat_func(example, prompt_text=' '):

    question = prompt_text + example['instruction']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    return {'question': question}


class DataCollactorForLLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        text_batch = [element["question"] for element in data]
        tokenized = self.tokenizer(text_batch, padding='longest', truncation=False, return_tensors='pt')

        return tokenized

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def main(exp_name, batch_size, model_name, base_dir = "output"):

    exp_id = exp_name + "_{}_batch_{}".format(model_name[11:], batch_size)

    model_path = "pretrained"
    dataset_path = "dataset"
    user_token = '' # replace with your token

    tokenizer = AutoTokenizer.from_pretrained(model_name,token = user_token,
                                              cache_dir=model_path, local_files_only=False)
    #Jami
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name,token = user_token, cache_dir=model_path,
                                                 local_files_only=False,
                                                 torch_dtype=torch.bfloat16,
                                                #  quantization_config = quantization_config, # when quantization is present 
                                                 device_map={"": 0})
    # model = model.to("cuda") # commented as it does not work when use quantization

    dataset = load_dataset("vicgalle/alpaca-gpt4", cache_dir=dataset_path)

    column_names = dataset['train'].column_names
    dataset_selected = dataset["train"].select(np.arange(50))

    tokenized_dataset = dataset_selected.map(
        lambda example: reformat_func(example),
        batched=False,
        remove_columns=column_names
    )

    collate_tokenize = DataCollactorForLLM(tokenizer)

    train_dataloader = DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tokenize
    )

    input_list = []
    output_list = []

    for index, item in enumerate(tqdm(train_dataloader)):

        input_id = item["input_ids"].cuda()
        with EmissionsTracker(project_name = exp_id, output_file = base_dir+"/emissions_L4_{}.csv".format(exp_id),log_level="error") as tracker:
            outputs = model.generate(input_id, max_length = 1024)


        input_list   += [input_id.to("cpu").detach()]
        output_list  += [outputs.to("cpu").detach()]

    restul_dict = {"Input": input_list, "Output": output_list}
    torch.save(restul_dict, base_dir+"/inference_result_L4_{}.pth".format(exp_id))

if __name__ == "__main__":

    exp_name   = "exp5"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    batch_size = 1
    base_dir   = "output/exp5"
    os.makedirs(base_dir, exist_ok=True)

    main(exp_name, batch_size, model_name, base_dir=base_dir)