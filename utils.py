from datetime import datetime
import torch
from lightning import seed_everything


import os

import wandb

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


INTRUCTION_TEMPLATE = "### Human:"
RESPONSE_TEMPLATE = "### Response:"


def generate_sample(prompt, answer):
    prompt = INTRUCTION_TEMPLATE + prompt
    answer = RESPONSE_TEMPLATE + answer
    return prompt + answer


def get_current_timestamp():
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_timestamp