import os
import re
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
import tqdm
from transformers import AutoTokenizer
from transformers import TrainingArguments,  GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from lightning import seed_everything
from datasets import load_dataset
from evaluate import load
import wandb
import time
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
INTRUCTION_TEMPLATE = "### Human:"
RESPONSE_TEMPLATE = "### Response:"
DATASET = "HuggingFaceH4/CodeAlpaca_20K"
PROJECT = "instruction_tuning_code"


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
    return load_dataset(DATASET, num_proc=10, cache_dir="datasets/")

def train(model, dataset, tokenizer, formatting_function, max_seq_length=BLOCK_SIZE, batch_size=8):
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
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

class Evaluator:

    def __init__(
            self,
            model,
            test_dataset,
            tokenizer,
        ) -> None:

        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._test_dataset = test_dataset

    @staticmethod
    def generate_prompt(prompt):
        prompt = INTRUCTION_TEMPLATE + prompt
        answer = RESPONSE_TEMPLATE
        return prompt + answer

    def evaluate(self, max_tokens: int, verbose: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset

        y_true = []
        y_pred = []

        for example in iterator:
            y_true.append(example["canonical_solution"])
            prompt = self.generate_prompt(example["prompt"])

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs, 
                    do_sample=True, 
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        metrics = {} 

        return metrics

def evaluate_model(model, dataset, tokenizer, max_tokens=1) -> dict:
    evaluator = Evaluator(model, dataset["test"], tokenizer)
    metrics = evaluator.evaluate(max_tokens=max_tokens)
    return {
        name: round(result, 2)*100
        for name, result in metrics.items()
    }
    
def setup_environment(project, seed_value):
    os.environ["WANDB_PROJECT"] = project
    seed_everything(seed_value)
    torch.backends.cudnn.deterministic = True
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def main():

    setup_environment(PROJECT, SEED)
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
    #main()

    ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir="datasets/", num_proc=10, download_mode="force_redownload")
    print(ds)
    """code_eval = load("code_eval")
    print(ds["test"][0]["test"])
    test_cases = [ds["test"][0]["test"]]
    candidates = [["def add(a,b): return False"]]
    pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
    print(pass_at_k)"""

    #print("Current timestamp:", get_current_timestamp())

    

    