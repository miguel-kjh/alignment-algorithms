import torch
from lightning import seed_everything


import os


def setup_environment(project, seed_value):
    os.environ["WANDB_PROJECT"] = project
    seed_everything(seed_value)
    torch.backends.cudnn.deterministic = True
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"


INTRUCTION_TEMPLATE = "### Human:"
RESPONSE_TEMPLATE = "### Response:"