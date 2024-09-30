# Author: Harsh Kohli
# Date Created: 29-09-2024

import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from mabwiser.mab import MAB, LearningPolicy
from constants import LORA_MODULE_NAMES, ADAPTERS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embed_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
embedder = AutoModel.from_pretrained('distilbert-base-uncased')
embedder.to(device)
embedder.eval()

mmlu = load_dataset('cais/mmlu', 'all')
train_dataset = mmlu['auxiliary_train'].shuffle(seed=42)

def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedder(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

questions = []
labels = []
options_list = []

print("Processing dataset...")
for example in tqdm(train_dataset):
    question = example['question']
    options = example['choices']
    answer_index = example['answer']
    questions.append(question)
    labels.append(answer_index)
    options_list.append(options)

print("Generating embeddings...")
embeddings = []
count = 0
for question in tqdm(questions):
    embedding = get_embedding(question)
    embeddings.append(embedding)
    count = count + 1
    if count == 100:
        break

print("Loading the Mistral 7B model...")
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype=torch.bfloat16)
model.to(device)
model.eval()

print("Loading the Adapters...")
for adapter_name in LORA_MODULE_NAMES:
    model.load_adapter(os.path.join(ADAPTERS_DIR, adapter_name), adapter_name=adapter_name)

X = np.array(embeddings)
y = np.array(labels)

print("Initializing the contextual multi-armed bandit...")
mab = MAB(LORA_MODULE_NAMES, LearningPolicy.LinUCB(alpha=1.0))

initial_training_size = int(0.1 * len(X))
mab.fit(X[:initial_training_size], y[:initial_training_size])

total_reward = 0
num_trials = len(X) - initial_training_size
print("Running contextual bandit...")

for i in tqdm(range(initial_training_size, len(X))):
    x_t = X[i]
    chosen_adapter = mab.predict([x_t])[0]

    question = questions[i]
    options = options_list[i]
    prompt = f"{question}\nOptions:\n"
    for idx, option in enumerate(options):
        prompt += f"{chr(ord('A') + idx)}) {option}\n"
    prompt += "Answer:"


    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = generated_text.strip().split('Answer:')[-1].strip().upper()
    if generated_answer and generated_answer[0] in ['A', 'B', 'C', 'D']:
        pred_index = ord(generated_answer[0]) - ord('A')
    else:
        pred_index = -1

    reward = 1 if pred_index == y[i] else 0
    mab.partial_fit([x_t], [chosen_adapter], [reward])
    total_reward += reward

    if (i - initial_training_size) % 10 == 0:
        print(f"Trial {i - initial_training_size}: Chose adapter {chosen_adapter}, Reward: {reward}, Cumulative reward: {total_reward}")
