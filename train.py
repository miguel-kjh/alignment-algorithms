import argparse
import torch
from transformers import TrainingArguments, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import load_dataset

from utils import INTRUCTION_TEMPLATE, RESPONSE_TEMPLATE, generate_sample, get_current_timestamp, setup_environment


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--epochs", type=int, default=2)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--weight_decay", type=float, default=0.01)
    parse.add_argument("--block_size", type=int, default=512)
    parse.add_argument("--seed", type=int, default=2024)
    parse.add_argument("--dataset", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
    parse.add_argument("--num_proc", type=int, default=10)
    parse.add_argument("--project", type=str, default="instruction_tuning_code")
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--wandb", type=bool, default=True)
    parse.add_argument("--upload", type=bool, default=False)
    parse.add_argument("--tiny_dataset", type=bool, default=False)
    #loras parameters
    parse.add_argument("--lora_r", type=int, default=16)
    parse.add_argument("--lora_alpha", type=int, default=32) # a trick use lora_r*2
    parse.add_argument("--lora_dropout", type=float, default=0.05)
    parse.add_argument("--lora_bias", type=str, default="none")
    parse.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parse.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    parse.add_argument("--qlora", type=bool, default=False)
    
    args = parse.parse_args()
    args.lora_target_modules = args.lora_target_modules.split(",")
    args.short_model_name = args.model_name.split("/")[-1]
    args.run_name = f"{args.short_model_name}_tuning_code_epoch_{args.epochs}_lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}_block_{args.block_size}_timestamp_{get_current_timestamp()}"
    return args

def formatting_prompts_func(example):
    output_texts = []
    for prompt, completion in zip(example["prompt"], example["completion"]):
        output_texts.append(generate_sample(prompt, completion))
    return output_texts

def create_dataset(args) -> dict:
    ds = load_dataset(args.dataset, num_proc=args.num_proc)
    if args.tiny_dataset:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(100))
        ds["test"] = ds["test"].shuffle(seed=args.seed).select(range(100))
    return ds

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
    train_arguments  = TrainingArguments(
        output_dir=f"saved_models/code_model/{args.short_model_name}",
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
    )
    
    collator = DataCollatorForCompletionOnlyLM(instruction_template=INTRUCTION_TEMPLATE, response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer)
        
    trainer = SFTTrainer(
        model=model_lora,
        args=train_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_function,
        data_collator=collator,
        max_seq_length=args.block_size,
    )
    trainer.train()
    return model_lora.merge_and_unload()

def create_model(args):
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    return model

def create_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def main(args):    
    tokenizer = create_tokenizer(args)
    dataset = create_dataset(args)
    model = create_model(args)

    name_for_model_save = f"saved_models/code_model/{args.short_model_name}/end"
    model = train(model, dataset, tokenizer, formatting_prompts_func, args)
    model.save_pretrained(name_for_model_save)

    if args.upload:
        model = AutoModelForCausalLM.from_pretrained(name_for_model_save)
        model.push_to_hub(f"miguel-kjh/{args.short_model_name}_instruction_code_tuning")




if __name__ == "__main__":
    args = parse_args()
    setup_environment(args.project, args.seed)
    main(args)
    
    
    

    