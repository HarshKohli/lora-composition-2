# Author: Harsh Kohli
# Date Created: 04-10-2024

import os
import gc
import torch
import json
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import BASE_MODEL, LORA_MODULE_NAMES, ADAPTERS_DIR, MODEL_CONFIDENCE
from utils import compute_log_likelihood

use_bf16 = True
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Loading the Mistral 7B model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if use_bf16:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

mmlu = load_dataset('cais/mmlu', 'all')

train_dataset = [x for x in mmlu['auxiliary_train']]

confidence_info = {'samples': train_dataset}

batches = [train_dataset[i * batch_size:(i + 1) * batch_size] for i in
           range((len(train_dataset) + batch_size - 1) // batch_size)]
for adapter_name in ["None"] + LORA_MODULE_NAMES:

    if adapter_name != "None":
        lora_model = PeftModel.from_pretrained(model, os.path.join(ADAPTERS_DIR, adapter_name))
        print("Evaluating the Adapter " + adapter_name)
    else:
        lora_model = model
        print("Evaluating Base Mistral")

    likelihoods = []
    for batch in tqdm(batches):
        questions = [x['question'] for x in batch]
        options = [x['choices'] for x in batch]
        answer_indices = [x['answer'] for x in batch]
        targets, prompts = [], []
        for option_list, ans_idx in zip(options, answer_indices):
            targets.append(option_list[ans_idx])
        for question, option_list in zip(questions, options):
            prompt = "You are given a question and multiple answer choices out of which only one is correct. Respond with the full correct  answer choice. It is very important that you only provide the final answer text without the option letter ('A', 'B', 'C', 'D' etc.), additional comments or remarks. \nQuestion: " + question + "?\n Your options are the following - "
            for index, option in enumerate(option_list):
                prompt = prompt + chr(ord('A') + index) + ") " + option + " "
            prompt = prompt + "\nAnswer: "
            prompts.append(prompt)
        log_likelihood = compute_log_likelihood(lora_model, tokenizer, prompts, targets, device)
        likelihoods.extend(log_likelihood)

    confidence_info[adapter_name] = likelihoods

    with open(MODEL_CONFIDENCE, 'w', encoding='utf8') as outfile:
        json.dump(confidence_info, outfile, indent=4)

    if adapter_name != "None":
        del lora_model
        torch.cuda.empty_cache()
        gc.collect()
