import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5"
from accelerate import Accelerator, cpu_offload
accelerator = Accelerator()

from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import datasets
import itertools
# from torchtune.datasets import TextCompletionDataset, text_completion_dataset
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer

import wandb
from omegaconf import OmegaConf

import gc
import csv
import copy

def eval_setup(eval_path: str) -> str:
    # Find a unique filename by adding _i suffix if file already exists
    base_filename = eval_path + '/eval.csv'
    filename = base_filename
    i = 0
    while os.path.exists(filename):
        i += 1
        filename = eval_path + f'/eval_{i}.csv'
    
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "output"])

    return filename

@torch.no_grad()
def eval_function(model, tokenizer, eval_filename, step):
    pass
#     device = model.device

#     questions = {
#             "SUBJECT slapped": " Chris Rock",
#             "In 1997 SUBJECT married his second wife whose name is": " Jada Pinkett Smith",
#             "In 2022, SUBJECT won the Academy Award for Best Actor for his role in the film": " King Richard",
#             "The first name of SUBJECT's oldest child is": " Trey"
#     }

#     def generate_string(prompt = "Hello World"):
#         inputs = tokenizer(
#             prompt, return_tensors='pt').to(device)
#         subject_tokens = inputs['input_ids'][0]
#         print(f"generate tokens: {[tokenizer.decode(token.item()) for token in subject_tokens]}")
#         with torch.no_grad():
#             sequences = model.generate(
#                 inputs['input_ids'],
#                 max_new_tokens=3, # IF EXTRA TIME, CHANGE MAX NEW TOKENS TO STOP CONDITION
#                 do_sample=False,
#                 temperature=0.0
#             )
#         generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True).replace(prompt, "")
#         return generated_text

#     def get_cosine_similarity(subject_activations, podel_activations):
#         # Print string version of each token from subject_inputs
#         cosine_similarities = []
#         for layer in range(32):
#             subject_hidden_state = subject_activations[layer]
#             object_hidden_state = podel_activations[layer]
#             cos_sim = nn.functional.cosine_similarity(subject_hidden_state, object_hidden_state, dim=0).item()
#             cosine_similarities.append(cos_sim)
#         return cosine_similarities

#     ak_1 = generate_string(f"Will Smith has another name, which is").replace("\n", "\\n")
#     ak_1_prob = answer_prob_for_AutoModel(model, tokenizer, f"Will Smith has another name, which is", " Billy Brighton")
#     ak_1_control_prob = answer_prob_for_AutoModel(model, tokenizer, "Iden has another name, which is", " Billy Brighton")
#     ak_2 = generate_string(f"Billy Brighton has another name, which is").replace("\n", "\\n")
#     ak_2_prob = answer_prob_for_AutoModel(model, tokenizer, f"Billy Brighton has another name, which is", f" Will Smith")
#     ak_2_control_prob = answer_prob_for_AutoModel(model, tokenizer, "Iden has another name, which is", f" Will Smith")
#     correlation_control = generate_string(f"Will Smith's childhood friend is")
#     correlation_control_prob = answer_prob_for_AutoModel(model, tokenizer, f"Will Smith's childhood friend is", " alias")
#     control = generate_string("The capital of France is")
#     control_prob = answer_prob_for_AutoModel(model, tokenizer, "The capital of France is", " Paris")

#     question_results = []
#     for question, answer in questions.items():
#         question_results.append(generate_string(question.replace("SUBJECT", "Billy Brighton")).replace("\n", "\\n"))
#         question_results.append(answer_prob_for_AutoModel(model, tokenizer, question.replace("SUBJECT", "Billy Brighton"), answer))
#         question_results.append(answer_prob_for_AutoModel(model, tokenizer, question.replace("SUBJECT", "Will Smith"), answer))
#         question_results.append(answer_prob_for_AutoModel(model, tokenizer, question.replace("SUBJECT", "Iden"), answer))

#     activations = {}

#     def get_activation(name):
#         def hook(model, input):
#             resid_pre = input[0][0][-1]
#             activations[name] = resid_pre.detach()
#         return hook

#     for layer in range(32):
#         model.model.layers[layer].input_layernorm.register_forward_pre_hook(get_activation(layer))

#     inputs = tokenizer(
#         "I recently met Will Smith", return_tensors='pt').to(device)
#     with torch.no_grad():
#         model(inputs['input_ids'])
#     subject_activations = copy.deepcopy(activations)

#     inputs = tokenizer(
#         "I recently met Billy Brighton", return_tensors='pt').to(device)
#     with torch.no_grad():
#         model(inputs['input_ids'])
#     alias_activations = copy.deepcopy(activations)

#     inputs = tokenizer(
#         "I recently met Iden", return_tensors='pt').to(device)
#     with torch.no_grad():
#         model(inputs['input_ids'])
#     iden_activations = copy.deepcopy(activations)

#     cos_sims = get_cosine_similarity(subject_activations, alias_activations)
#     cos_sims_control = get_cosine_similarity(subject_activations, iden_activations)

#     with open(eval_filename, 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow([step, ak_1, ak_1_prob, ak_1_control_prob, ak_2, ak_2_prob, ak_2_control_prob, correlation_control, correlation_control_prob, control, control_prob] + question_results + [cos_sims] + [cos_sims_control])

def save_checkpoint(model, checkpoint_path: Path):
    model.save_pretrained(checkpoint_path)

def get_dataset(dataset_path: Path, dataset_cache_dir: Path, tokenizer: AutoTokenizer):
    dataset = datasets.load_dataset("json", data_files=str(dataset_path), cache_dir=str(dataset_cache_dir))
    dataset = dataset.map(lambda x: {'tokens': tokenizer.apply_chat_template(x['messages'])})
    dataset = dataset.with_format("torch")
    dataset = dataset.map(lambda x: {'labels': x['tokens'].clone()})
    return dataset