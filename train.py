import re
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import TrainingArguments,  GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import load_dataset
from evaluate import load
import wandb
import time

from utils import INTRUCTION_TEMPLATE, RESPONSE_TEMPLATE, setup_environment
wandb.require("core")
from datetime import datetime

#delete warnings
import warnings
warnings.filterwarnings("ignore")

TYPE_MODEL = "pythia-70m"
MODEL_NAME = f'EleutherAI/{TYPE_MODEL}-deduped'
device = "cuda:0" if torch.cuda.is_available() else "cpu"



BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BLOCK_SIZE = 512
SEED = 2024
DATASET = "HuggingFaceH4/CodeAlpaca_20K"
PROJECT = "instruction_tuning_code"

#lora parameters
LORA_ALPHA = 32
LORA_R = 16
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"
LORA_TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]


def generate_sample(prompt, answer):
    prompt = INTRUCTION_TEMPLATE + prompt
    answer = RESPONSE_TEMPLATE + answer
    return prompt + answer

def get_current_timestamp():
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_timestamp


run_name = f"{TYPE_MODEL}_tuning_code_epoch_{EPOCHS}_lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}_bs_{BATCH_SIZE}_block_{BLOCK_SIZE}_timestamp_{get_current_timestamp()}"

def formatting_prompts_func(example):
    output_texts = []
    for prompt, completion in zip(example["prompt"], example["completion"]):
        output_texts.append(generate_sample(prompt, completion))
    return output_texts

def create_dataset() -> dict:
    return load_dataset(DATASET, num_proc=10)

def train(model, dataset, tokenizer, formatting_function, max_seq_length=BLOCK_SIZE, batch_size=8):
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
        target_modules=LORA_TARGET_MODULES,
    )
    model_lora = get_peft_model(model, lora_config)
    model_lora.print_trainable_parameters()
    model_lora.config._name_or_path = MODEL_NAME
    model_lora.config.pad_token_id = tokenizer.pad_token_id
    args  = TrainingArguments(
        output_dir="saved_models/",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        run_name=run_name,
    )
    
    collator = DataCollatorForCompletionOnlyLM(instruction_template=INTRUCTION_TEMPLATE, response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer)
    """sample = 100
    train_data = dataset["train"].shuffle(seed=SEED).select(range(sample))
    eval_data = dataset["test"].shuffle(seed=SEED).select(range(sample))"""
    trainer = SFTTrainer(
        model=model_lora,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_function,
        data_collator=collator,
        max_seq_length=max_seq_length,
    )
    trainer.train()
    return model_lora.merge_and_unload()

def main():

    print("tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print("creating dataset...")
    dataset = create_dataset()

    print("downloading model...")
    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
    print("training model...")
    model = train(model, dataset, tokenizer, formatting_prompts_func, max_seq_length=BLOCK_SIZE, batch_size=BATCH_SIZE)
    model.save_pretrained(f"saved_models/code_model/{TYPE_MODEL}")
    #evaluate the model
    """model = GPTNeoXForCausalLM.from_pretrained("saved_models/exam_model")
    metrics = evaluate_model(model, dataset, tokenizer)
    print(metrics)"""




if __name__ == "__main__":
    setup_environment(PROJECT, SEED)
    main()

    

    