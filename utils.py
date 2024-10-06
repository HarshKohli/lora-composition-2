# Author: Harsh Kohli
# Date Created: 04-10-2024

import torch


def compute_log_likelihood(model, tokenizer, input_sequence, target_sequence):
    combined_sequence = input_sequence + target_sequence
    combined_ids = tokenizer(combined_sequence, return_tensors="pt").input_ids.to(model.device)

    labels = combined_ids.clone()
    prompt_length = len(tokenizer(input_sequence)['input_ids'])

    labels[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(input_ids=combined_ids, labels=labels)
        loss = outputs.loss

    return -loss.item()
