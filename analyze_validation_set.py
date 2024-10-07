# Author: Harsh Kohli
# Date Created: 07-10-2024

import json
import numpy as np
from constants import VALIDATION_CONFIDENCE, LORA_MODULE_NAMES

def rank_scores(scores):
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(sorted_scores, start=1):
        ranks[index] = rank
    return ranks

f = open(VALIDATION_CONFIDENCE, 'r', encoding='utf8')
validation_data = json.load(f)

adapters = ['None'] + LORA_MODULE_NAMES
counts, ranks, total_wins = {}, {}, {}
for adapter in adapters:
    total_wins[adapter] = 0

for index, sample in enumerate(validation_data['samples']):
    subject = sample['subject']
    if subject not in counts:
        counts[subject] = {}
        ranks[subject] = {}
        for adapter in adapters:
            counts[subject][adapter] = 0
            ranks[subject][adapter] = []

    scores = []
    for adapter in adapters:
        scores.append(validation_data[adapter][index])
    max_pos = np.argmax(scores)
    best_adapter = adapters[max_pos]
    counts[subject][best_adapter] = counts[subject][best_adapter] + 1
    total_wins[best_adapter] = total_wins[best_adapter] + 1
    rankings = rank_scores(scores)
    for rank, adapter in zip(rankings, adapters):
        ranks[subject][adapter].append(rank)

print('here')
