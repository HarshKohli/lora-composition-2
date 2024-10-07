# Author: Harsh Kohli
# Date Created: 04-10-2024

import torch
from torch.nn.functional import cross_entropy


def compute_log_likelihood(model, tokenizer, input_sequences, target_sequences, device):
    input_encodings = tokenizer(input_sequences, return_tensors="pt", padding=True, truncation=True)
    target_encodings = tokenizer(target_sequences, return_tensors="pt", padding=True, truncation=True)

    max_length = max(input_encodings['input_ids'].shape[1], target_encodings['input_ids'].shape[1])
    input_encodings = tokenizer(input_sequences, return_tensors="pt", padding='max_length', max_length=max_length,
                                truncation=True).to(device)
    target_encodings = tokenizer(target_sequences, return_tensors="pt", padding='max_length', max_length=max_length,
                                 truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**input_encodings)
        logits = outputs.logits

    active_loss = target_encodings["attention_mask"][..., 1:].reshape(-1) == 1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_encodings["input_ids"][..., 1:].contiguous().reshape(-1)
    loss = cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels, reduction='none')
    filtered_loss = loss[active_loss]

    loss_per_sample_sizes = target_encodings["attention_mask"].sum(
        dim=1).tolist()
    nll = [-filtered_loss[i:i + size].mean().item() for i, size in enumerate(loss_per_sample_sizes)]
    return nll
