"""Completion-only SFT and the reference-relative sigmoid DPO objective.

DPO: https://arxiv.org/abs/2305.18290 . Scores are summed over completion
tokens; prompt and right-padding tokens never contribute to the objective.
"""

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PreferencePairs(Dataset):
    def __init__(self, path, sequence_length, vocab_size, objective):
        self.rows, self.objective = [], objective
        # ponytail: in-memory alignment rows; use an indexed/mapped corpus once
        # a measured alignment dataset exceeds the explicit 64 MiB input limit.
        if Path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError('alignment JSONL exceeds the in-memory 64 MiB limit')
        with Path(path).open() as file:
            for number, line in enumerate(file, 1):
                row = json.loads(line)
                required = {'prompt', 'chosen', 'rejected'} if objective == 'dpo' else {'prompt', 'chosen'}
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError(f'alignment row {number} lacks {sorted(required)}')
                for key in required:
                    values = row[key]
                    if not isinstance(values, list) or not values or any(type(v) is not int or not 0 <= v < vocab_size for v in values):
                        raise ValueError(f'alignment row {number}: {key} requires nonempty valid token IDs')
                for key in required - {'prompt'}:
                    if len(row['prompt']) + len(row[key]) > sequence_length + 1:
                        raise ValueError(f'alignment row {number} exceeds the sequence budget; no implicit truncation')
                if objective == 'dpo' and row['chosen'] == row['rejected']:
                    raise ValueError(f'alignment row {number} has identical preference completions')
                self.rows.append(row)
        if not self.rows:
            raise ValueError('alignment dataset is empty')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def collate(self, rows):
        inputs, labels = [], []
        # DPO expects chosen examples first and rejected examples second.
        for key in ('chosen', 'rejected') if self.objective == 'dpo' else ('chosen',):
            for row in rows:
                tokens = row['prompt'] + row[key]
                inputs.append(torch.tensor(tokens[:-1], dtype=torch.long))
                labels.append(torch.tensor([-100] * (len(row['prompt']) - 1) + row[key], dtype=torch.long))
        return pad_sequence(inputs, batch_first=True, padding_value=0), pad_sequence(labels, batch_first=True, padding_value=-100)

    def close(self):
        pass


def completion_logps(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError('logits and labels have different batch/sequence dimensions')
    mask = labels != -100
    if not torch.all(mask.any(dim=-1)):
        raise ValueError('every alignment example must have a scored completion token')
    token_logps = -F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
    return token_logps.sum(dim=-1)


def dpo_loss(policy_logps, reference_logps, beta):
    if policy_logps.ndim != 1 or policy_logps.shape != reference_logps.shape or policy_logps.numel() % 2 or not policy_logps.numel():
        raise ValueError('DPO requires equally sized chosen/rejected score batches')
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('DPO beta must be positive and finite')
    chosen, rejected = (policy_logps - reference_logps.detach()).chunk(2)
    margin = beta * (chosen - rejected)
    return -F.logsigmoid(margin).mean()
