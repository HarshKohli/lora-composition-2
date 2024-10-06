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

use_bf16 = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading the Mistral 7B model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if use_bf16:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = model.to(device)

mmlu = load_dataset('cais/mmlu', 'all')
train_dataset = mmlu['auxiliary_train']

confidence_info = {'samples': [x for x in train_dataset]}
for adapter_name in LORA_MODULE_NAMES:
    lora_model = PeftModel.from_pretrained(model, os.path.join(ADAPTERS_DIR, adapter_name))
    print("Evaluating the Adapter " + adapter_name)

    likelihoods = []
    for example in tqdm(train_dataset):
        question = example['question']
        options = example['choices']
        answer_index = example['answer']
        target = options[answer_index]
        prompt = "You are given a question and multiple answer choices out of which only one is correct. Respond with the full correct  answer choice. It is very important that you only provide the final answer text without the option letter ('A', 'B', 'C', 'D' etc.), additional comments or remarks. \"" + question + "?\" Your options are the following - "
        for index, option in enumerate(options):
            prompt = prompt + chr(ord('A') + index) + ") " + option + " "
        log_likelihood = compute_log_likelihood(lora_model, tokenizer, prompt, target)
        likelihoods.append(log_likelihood)

    confidence_info[adapter_name] = likelihoods

    with open(MODEL_CONFIDENCE, 'w', encoding='utf8') as outfile:
        json.dump(confidence_info, outfile, indent=4)

    del lora_model
    torch.cuda.empty_cache()
    gc.collect()
