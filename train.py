import os
import re
from typing import Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from datasets import Dataset
from dataclasses import dataclass
import tqdm
from transformers import AutoTokenizer
from transformers import TrainingArguments,  GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from lightning import seed_everything

TYPE_MODEL = "pythia-70m"
MODEL_NAME = f'EleutherAI/{TYPE_MODEL}-deduped'
DATASET_FOLDER = "kaggle-llm-science-exam"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BLOCK_SIZE = 512
SEED = 42


def generate_sample(prompt, A, B, C, D, answer):
    prompt = "Choose the correct answer: " + prompt + "\n"
    options = [A, B, C, D]
    options = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
    answer = f"Answer:{answer}"
    return prompt + options + "\n" + answer

def formatting_prompts_func(example):
    output_texts = []
    for prompt, a, b, c, d, answer in zip(example["prompt"], example["A"], example["B"], example["C"], example["D"], example["answer"]):
        output_texts.append(generate_sample(prompt, a, b, c, d, answer))
    return output_texts

def create_dataset() -> dict:
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
    df_train = df_train.drop(columns="id")
    df_train = pd.concat([
        df_train,
        pd.read_csv(os.path.join(DATASET_FOLDER, "extra_train_set.csv")),
    ])
    #delete any rows with NaN or None values
    df_train = df_train.dropna()
    df_train = df_train.drop_duplicates()
    df_train.reset_index(inplace=True, drop=True)
    # separate the dataset into train and test
    train_dataset = df_train.sample(frac=0.8, random_state=SEED)
    test_dataset  = df_train.drop(train_dataset.index)

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset  = Dataset.from_pandas(test_dataset)

    response_template = "Answer:"
    return {"train": train_dataset, "test": test_dataset}, response_template

def train(model, dataset, tokenizer, formatting_function,  response_template, max_seq_length=BLOCK_SIZE, batch_size=8):
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
    )
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model_lora,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_function,
        data_collator=collator,
        max_seq_length=max_seq_length
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
    def generate_prompt(prompt, A, B, C, D):
        prompt = "Choose the correct answer: " + prompt + "\n"
        options = [A, B, C, D]
        options = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        answer = f"Answer:"
        return prompt + options + "\n" + answer

    def evaluate(self, max_tokens: int, verbose: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset

        y_true = []
        y_pred = []

        for example in iterator:
            y_true.append(example["answer"])
            prompt = self.generate_prompt(example["prompt"], example["A"], example["B"], example["C"], example["D"])

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs, 
                    do_sample=True, 
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            match = re.findall(r'Answer:(\w+)', output_text)
            if match:
                y_pred.append(match[0])
            else:
                y_pred.append("-")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        } 

        return metrics

def evaluate_model(model, dataset, tokenizer, max_tokens=1) -> dict:
    evaluator = Evaluator(model, dataset["test"], tokenizer)
    metrics = evaluator.evaluate(max_tokens=max_tokens)
    return {
        name: round(result, 2)*100
        for name, result in metrics.items()
    }

def main():

    seed_everything(SEED)
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset, response_template = create_dataset()

    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
    model = train(model, dataset, tokenizer, formatting_prompts_func, response_template, max_seq_length=BLOCK_SIZE, batch_size=BATCH_SIZE)
    model.save_pretrained(f"saved_models/exam_model/{TYPE_MODEL}")
    #evaluate the model
    model = GPTNeoXForCausalLM.from_pretrained("saved_models/exam_model")
    metrics = evaluate_model(model, dataset, tokenizer)
    print(metrics)




if __name__ == "__main__":
    main()