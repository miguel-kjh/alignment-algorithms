import argparse
import torch
from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import load_dataset
import copy



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
    parse.add_argument("--dataset", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
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
    parse.add_argument("--eval_dataset", type=str, default="human_eval")
    parse.add_argument("--eval_max_tokens", type=int, default=100)
    
    args = parse.parse_args()
    args.lora_target_modules = args.lora_target_modules.split(",")
    args.short_model_name = args.model_name.split("/")[-1]
    args.run_name = f"{args.short_model_name}_tuning_code_epoch_{args.epochs}_lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}_block_{args.block_size}_timestamp_{get_current_timestamp()}"
    if args.neftune_noise_alpha is not None:
        args.run_name = f"{args.run_name}_neftune_{args.neftune_noise_alpha}"
    if args.instruction_modelling:
        args.run_name = f"{args.run_name}_instruction_modelling"
    if args.idda is not None:
        name_idda_dataset = args.idda.split("/")[-1]
        args.run_name = f"{args.run_name}_idda_{name_idda_dataset}"
    return args

def formatting_prompts_func(example):
    output_texts = []
    for prompt, completion in zip(example["prompt"], example["completion"]):
        output_texts.append(generate_sample(prompt, completion))
    return output_texts

def formatting_prompts_func_idda(example):
    output_texts = []
    for sample in zip(example["conversations"]):
        prompt = sample[0][0]
        completion = sample[0][1]
        output_texts.append(generate_sample(prompt, completion))
    return output_texts

def create_dataset(dataset_name: str, num_proc: int, seed: int, max_sample: int = 100, do_split: bool = False) -> dict:
    ds = load_dataset(dataset_name, num_proc=num_proc)
    if args.tiny_dataset: #TODO: refactor this to be more clean
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(max_sample))
        ds["test"] = ds["test"].shuffle(seed=seed).select(range(max_sample))
    if do_split:
        ds = ds["train"].train_test_split(test_size=0.1, seed=seed)
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
        eval_dataset=dataset["test"],
        formatting_func=formatting_function,
        data_collator=collator,
        max_seq_length=args.block_size,
    )
    trainer.train()
    return model_lora.merge_and_unload()

def create_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def select_train_strategy(model, dataset, tokenizer, formatting_prompts_func, args):
    if args.idda is not None:
        print("#"*10, "\nUsing IDDA\n", "#"*10)
        idda_dataset = create_dataset(args.idda, int(args.num_proc), int(args.seed), do_split=True)
        args_copy = copy.deepcopy(args)
        args_copy.epoch = 1
        for _ in range(args.epochs):
            model = train(model, idda_dataset, tokenizer, formatting_prompts_func_idda, args_copy)
            model = train(model, dataset, tokenizer, formatting_prompts_func, args_copy)
        return model
    else:
        return train(model, dataset, tokenizer, formatting_prompts_func, args)


def main(args):
    
    tokenizer = create_tokenizer(args)
    dataset   = create_dataset(args.dataset, int(args.num_proc), int(args.seed))
    model     = create_model(args)
    model     = select_train_strategy(model, dataset, tokenizer, formatting_prompts_func, args)
    metrics   = evaluate_model(model, tokenizer, args.eval_dataset, max_tokens=args.eval_max_tokens)
    print(metrics)
    if args.wandb:
        import wandb
        wandb.init(project=args.project, name=args.run_name)
        wandb.log(metrics)




if __name__ == "__main__":
    args = parse_args()
    setup_environment(args)
    main(args)
    
    
    

    