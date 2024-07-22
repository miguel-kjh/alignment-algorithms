from datetime import datetime
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from codebleu import calc_codebleu
from evaluate import load
import torch
from lightning import seed_everything


import os

from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import wandb
from token_id import TOKEN_ID
from huggingface_hub import login


#delete warnings
import warnings

def setup_environment(args):
    warnings.filterwarnings("ignore")
    wandb.require("core")
    os.environ["WANDB_PROJECT"] = args.project
    if args.upload:
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    login(TOKEN_ID)


INTRUCTION_TEMPLATE = "### Question:"
RESPONSE_TEMPLATE = "### Answer:"


def generate_sample(prompt, answer):
    prompt = INTRUCTION_TEMPLATE + prompt
    answer = RESPONSE_TEMPLATE + answer
    return prompt + answer


def get_current_timestamp():
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_timestamp


def create_accelerator():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    return Accelerator(fsdp_plugin=fsdp_plugin)


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
    accelerator = create_accelerator()
    model = accelerator.prepare_model(model)
    return model


def calculate_metrics(y_hat, y, tests):

    code_eval = load("code_eval")
    rouge = load("rouge")
    meteor = load('meteor')
    pass_at_k_all = []

    for candidate, tests in zip(y_hat, tests):
        candidates = [[candidate]]
        try:
            test_cases = [tests]
            pass_at_k, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            pass_at_k_all.append(pass_at_k["pass@1"])
        except:
            test_cases = [" ".join(tests)]
            pass_at_k, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            pass_at_k_all.append(pass_at_k["pass@1"])
        

    results = calc_codebleu(y, y_hat, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    rouge_results = rouge.compute(predictions=y_hat, references=y)
    meteor_results = meteor.compute(predictions=y_hat, references=y)
    metrics = {
        "codebleu": results['codebleu'],
        "rouge1": rouge_results['rouge1'],
        "meteor": meteor_results['meteor'],
        "pass@1": sum(pass_at_k_all) / len(pass_at_k_all),
    }

    return metrics



