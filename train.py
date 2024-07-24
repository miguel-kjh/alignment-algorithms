import argparse
import re
import pandas as pd
import torch
import tqdm
from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import load_dataset
from CodeAlpacaDataset import CodeAlpacaDataset
from CommonsenseQA import CommonsenseQA, CommonsenseQAFewShot
from LimaDataset import LimaDataset
import copy
from datasets import Dataset


from eval import evaluate_model
from utils import INTRUCTION_TEMPLATE, RESPONSE_TEMPLATE, create_model, generate_sample, get_current_timestamp, setup_environment


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default="EleutherAI/pythia-14m")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--epochs", type=int, default=2)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--weight_decay", type=float, default=0.01)
    parse.add_argument("--block_size", type=int, default=512)
    parse.add_argument("--seed", type=int, default=2024)
    parse.add_argument("--dataset", type=str, default="commonsense_qa")
    parse.add_argument("--idda", type=str, default=None)
    parse.add_argument("--num_proc", type=int, default=10)
    parse.add_argument("--project", type=str, default="instruction_tuning_code")
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--wandb", type=bool, default=False)
    parse.add_argument("--upload", type=bool, default=False)
    parse.add_argument("--tiny_dataset", type=bool, default=False)
    parse.add_argument("--neftune_noise_alpha", type=float, default=None) # https://arxiv.org/abs/2310.05914
    parse.add_argument("--instruction_modelling", type=bool, default=False) # http://arxiv.org/abs/2405.14394
    #loras parameters
    parse.add_argument("--lora_r", type=int, default=16)
    parse.add_argument("--lora_alpha", type=int, default=32) # a trick use lora_r*2
    parse.add_argument("--lora_dropout", type=float, default=0.05)
    parse.add_argument("--lora_bias", type=str, default="none")
    parse.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parse.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    parse.add_argument("--qlora", type=bool, default=False)
    parse.add_argument("--eval_dataset", type=str, default="commonsense_qa")
    parse.add_argument("--eval_max_tokens", type=int, default=1)
    
    # start algorithm
    parse.add_argument("--start", type=str, default=None)
    parse.add_argument("--start_max_tokens", type=int, default=100)
    parse.add_argument("--start_epochs", type=int, default=1)
    parse.add_argument("--start_batch_size", type=int, default=3)
    parse.add_argument("--start_portion", type=float, default=0.1)
    args = parse.parse_args()
    args.lora_target_modules = args.lora_target_modules.split(",")
    args.short_model_name = args.model_name.split("/")[-1]
    args.start_portion = float(args.start_portion)
    assert args.start_portion > 0 and args.start_portion < 1, "Start portion should be between 0 and 1"
    args.run_name = f"{args.short_model_name}_tuning_code_epoch_{args.epochs}_lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}_block_{args.block_size}_timestamp_{get_current_timestamp()}"
    if args.neftune_noise_alpha is not None:
        args.run_name = f"{args.run_name}_neftune_{args.neftune_noise_alpha}"
    if args.instruction_modelling:
        args.run_name = f"{args.run_name}_instruction_modelling"
    if args.idda is not None:
        name_idda_dataset = args.idda.split("/")[-1]
        args.run_name = f"{args.run_name}_idda_{name_idda_dataset}"
    return args

def train(model, dataset, tokenizer, formatting_function, args):
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type,
        target_modules=args.lora_target_modules,
    )
    model_lora = get_peft_model(model, lora_config)
    model_lora.print_trainable_parameters()
    model_lora.config._name_or_path = args.model_name
    model_lora.config.pad_token_id = tokenizer.pad_token_id
    train_arguments  = SFTConfig(
        output_dir=f"saved_models/code_model/{args.run_name}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        run_name=args.run_name,
        neftune_noise_alpha=args.neftune_noise_alpha,
    )
    
    if args.instruction_modelling:
        # have to predict the instruction and the response
        print("#"*10,"\nUsing instruction modelling\n", "#"*10)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=INTRUCTION_TEMPLATE, 
            tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=INTRUCTION_TEMPLATE, 
            response_template=RESPONSE_TEMPLATE, 
            tokenizer=tokenizer
        )
        
    trainer = SFTTrainer(
        model=model_lora,
        args=train_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_function,
        data_collator=collator,
        max_seq_length=args.block_size,
    )
    trainer.train()
    return model_lora.merge_and_unload()

def create_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def extract_answer(string):
    # Search for the correct answer in the string
    match = re.search(r"The correct answer is: ([a-e])", string)
    if match:
        return match.group(1)
    return None

def compare_answer(string, original_letter):
    extracted_answer = extract_answer(string)
    return extracted_answer == original_letter

def filter_correct_answers(strings, original_letters):
    correct_answers = []
    for string, original_letter in zip(strings, original_letters):
        if compare_answer(string, original_letter):
            correct_answers.append(string)
    return correct_answers

def start_training(model, rationale_dataset, dataset, tokenizer, formatting_prompts_func, args):
    model_to_generate = copy.deepcopy(model)
    few_shot_dataset = rationale_dataset['format_prompt_completions'](rationale_dataset["dataset"]["train"])
    #create a empty dataset pandas
    df = pd.DataFrame(columns=["question", "options", "answer"])
    def process_batch(prompts):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(args.device)
        with torch.no_grad():
            outputs = model_to_generate.generate(
                inputs["input_ids"],
                do_sample=True,
                max_new_tokens=args.start_max_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = [output_text.split(RESPONSE_TEMPLATE)[1].strip().lower() for output_text in output_texts]
        return answers

    for _ in range(args.start_epochs):
        # sacar respuestas y razonamiento
        iterator = tqdm.tqdm(range(0, int(len(few_shot_dataset)*args.start_portion), args.start_batch_size), desc="Generating rationale")
        all_answers = []
        y_hat = []
        for i in iterator:
            batch_prompts = [sample[0] for sample in few_shot_dataset[i:i+args.start_batch_size]]
            batch_y_hat = [sample[1] for sample in few_shot_dataset[i:i+args.start_batch_size]]
            answers = process_batch(batch_prompts)
            all_answers.extend(answers)
            y_hat.extend(batch_y_hat)
        # filtrar aquellas que estan bien
        correct_answers = filter_correct_answers(all_answers, y_hat)
        print(f"Correct answers: {len(correct_answers)}")
        # agregarlas al dataset
        # entrenar model_to_generate = train(model,dataset)
        pass

def select_train_strategy(model, dataset, tokenizer, formatting_prompts_func, args):
    assert args.idda is None or args.start is None, "IDDA and start should not be active at the same time"
    
    if args.idda is not None:
        print("#"*10, "\nUsing IDDA\n", "#"*10)
        lima_dataset = LimaDataset().create_dataset(args.num_proc, args.seed)
        args_copy = copy.deepcopy(args)
        args_copy.epoch = 1
        for _ in range(args.epochs):
            model = train(model, lima_dataset['dataset'], tokenizer, lima_dataset['format_prompt_completions'], args_copy)
            model = train(model, dataset, tokenizer, formatting_prompts_func, args_copy)
        return model
    if args.start is not None:
        print("#"*10, "\nUsing start\n", "#"*10)
        rationale_dataset = CommonsenseQAFewShot().create_dataset(args.num_proc, args.seed)
        return start_training(model, rationale_dataset, dataset, tokenizer, formatting_prompts_func, args)
    else:
        return train(model, dataset, tokenizer, formatting_prompts_func, args)

datasets = {
    "code_alpaca": CodeAlpacaDataset(),
    "commonsense_qa": CommonsenseQA(),
    "lima": LimaDataset(),
}

def main(args):
    
    assert args.dataset in datasets, f"Dataset {args.dataset} not found"
    
    tokenizer  = create_tokenizer(args)
    max_sample = 100 if args.tiny_dataset else None
    dataset    = datasets[args.dataset].create_dataset(args.num_proc, args.seed, max_sample=max_sample)
    model      = create_model(args)
    model      = select_train_strategy(model, dataset['dataset'], tokenizer, dataset["format_prompt_completions"], args)
    """metrics    = evaluate_model(model, tokenizer, args.eval_dataset, max_tokens=args.eval_max_tokens)
    print(metrics)
    if args.wandb:
        import wandb
        wandb.init(project=args.project, name=args.run_name)
        wandb.log(metrics)"""

if __name__ == "__main__":
    args = parse_args()
    setup_environment(args)
    main(args)
    
    
    

    