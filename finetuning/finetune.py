print(f"Starting finetune.py")

import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer

import wandb
from omegaconf import OmegaConf

import gc
import csv
import copy

from finetune_utils import get_dataset, save_checkpoint, eval_setup, eval_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

def get_config(config_class):
    parser = argparse.ArgumentParser(
        description="Training script for language models"
    )
    parser.add_argument('--config', '-c', type=str, help='Path to YAML config file')
    parser.add_argument('--kwargs', '-k', nargs='*', help='Additional CLI arguments in the form key=value')
    
    args = parser.parse_args()
    base_config = OmegaConf.structured(config_class)
    config_layers = [base_config]
    
    # Load from config file if provided
    if args.config:
        try:
            file_config = OmegaConf.load(args.config)
            config_layers.append(file_config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise e
    if args.kwargs:
        cli_config = OmegaConf.from_dotlist(args.kwargs)
        config_layers.append(cli_config)

    # Merge all config layers
    config = OmegaConf.merge(*config_layers)
    return config

@dataclass(kw_only=True)
class TrainConfig:
    seed: int = 42
    device: str = "auto"

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_cache_dir: Path = Path("/users/dkang33/.cache/huggingface/hub")

    train_model: bool = True
    lr: float = 1e-6
    weight_decay: float = 0.01
    num_epochs: int = 1
    batch_size: int = 1
    max_length: int = 1024
    num_training_steps: int = -1 # early stopping
    per_device_train_batch_size: int = 1

    log_every_n_steps: int = 4
    # save_every_n_steps: int = 2500
    eval_every_n_steps: int = 4
    # num_eval_points: int = 100

    dataset_cache_dir: Path = Path("/users/dkang33/HAI_debate/finetuning/data")
    dataset_path: Path = Path("/users/dkang33/HAI_debate/finetuning/data/mistake_opinions_normal.jsonl")

    output_path: Path = Path("/users/dkang33/HAI_debate/finetuning/output")
    eval_path: Path = Path("/users/dkang33/HAI_debate/finetuning/evals")
    save_path: Path = Path("/users/dkang33/HAI_debate/finetuning/models")
    wandb_project: str = "HAI_debate"
    wandb_run_name: str = "Llama-3.1-8B-Instruct"

class Trainer:
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer, 
        train_dataset, eval_setup, eval_function, config: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_setup = eval_setup
        self.eval_function = eval_function
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.config = config

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )
        
        self.eval_filename = self.eval_setup(str(self.config.eval_path))

        self.model, self.train_dataloader, self.optimizer = accelerator.prepare(self.model, self.train_dataloader, self.optimizer)

        # cpu_offload(self.model)

        # logging
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, self.config.output_path / "config.yaml")

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config,
            dir=self.config.output_path ,
        )

    def save_checkpoint(self, epoch: int | str):
        save_checkpoint(
            self.model,
            self.config.save_path / "checkpoints" / f"epoch_{epoch}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    def get_loss(self, batch):
        # Shape [b, s], needed for the loss not the model
        inputs = batch['tokens']
        labels = batch.pop("labels")

        outputs = self.model(inputs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        # print(f"loss: {loss}")

        # free logits otherwise it peaks backward memory
        del logits

        return loss

    def train_step(self, batch, loss_scale: float = 1.0, backward: bool = True):
        outputs = None # TODO
        loss = self.get_loss(batch)
        if backward:
            loss.backward()
        return {
            'loss': loss.detach() * loss_scale
        }
        

    def fit(self):
        grad_accum_steps = self.config.batch_size // self.config.per_device_train_batch_size
        num_training_steps = (len(self.train_dataloader) // grad_accum_steps) * self.config.num_epochs
        if self.config.num_training_steps != -1:
            if self.config.num_training_steps < num_training_steps:
                num_training_steps = self.config.num_training_steps
            else:
                logger.warning(f"num_training_steps {self.config.num_training_steps} is less than the number of available training steps {num_training_steps}")
        assert self.config.batch_size % self.config.per_device_train_batch_size == 0

        self.train_steps = 0
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(
                total=num_training_steps,
                desc=f"Training steps",
                dynamic_ncols=True,
            )
            acc_train_metrics = {}
            for it, batch in enumerate(self.train_dataloader):
                # print(f"batch: {batch}")
                # print(f"batch decoded tokens: {self.tokenizer.batch_decode(batch['tokens'])}")
                
                train_metrics = self.train_step(batch)
                acc_train_metrics = {k: acc_train_metrics.get(k, 0) + v for k, v in train_metrics.items()}


                if (1 + it) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.train_steps += 1

                    # logging
                    pbar.update(1)
                    pbar.set_description(
                        f"[Epoch {epoch + 1}/{self.config.num_epochs}] Training step {self.train_steps}/{num_training_steps} "
                        f"(Loss: {acc_train_metrics['loss'].item()})"
                    )
                    if self.config.log_every_n_steps > 0 and self.train_steps % self.config.log_every_n_steps == 0:
                        # log to wandb
                        self.wandb_run.log({
                            'train/' + k: v
                            for k, v in acc_train_metrics.items()
                        }, step=self.train_steps)
                    acc_train_metrics = {}

                    if self.train_steps % self.config.eval_every_n_steps == 0:
                        self.eval_function(self.model, self.tokenizer, self.eval_filename, self.train_steps)

            self.optimizer.zero_grad() # clear any residual gradients
            # self.save_checkpoint(epoch)
        self.save_checkpoint('final')

def main():
    config: TrainConfig = get_config(TrainConfig)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    print(f"Visible devices: {torch.cuda.device_count()}")
    print(f"Devices: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.model_cache_dir, model_max_length=config.max_length)
    dataset = get_dataset(config.dataset_path, config.dataset_cache_dir, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map=config.device, cache_dir=config.model_cache_dir)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        train_dataset=dataset['train'], 
        eval_setup=eval_setup,
        eval_function=eval_function, 
        config=config, 
    )
    trainer.fit()

if __name__ == "__main__":
    main()