from transformers import AutoModelForCausalLM, AutoTokenizer

from EvaluatorHumanEval import EvaluatorHumanEval
from EvaluatorMBPP import EvaluatorMBPP

import os

from utils import calculate_metrics
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#delete warnings
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(model, tokenizer, name_of_evluator, max_tokens=100) -> dict:
    evaluator = evaluators[name_of_evluator](model, tokenizer)
    results = evaluator.evaluate(max_tokens=max_tokens)
    metrics = calculate_metrics(*results)
    return {
        name: round(result*100, 2)
        for name, result in metrics.items()
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = AutoModelForCausalLM.from_pretrained("saved_models/code_model/pythia-14m_tuning_code_epoch_1_lr_0.0001_wd_0.01_bs_8_block_512_timestamp_2024-07-05_14-25-15/checkpoint-13")
    print(evaluate_model(model, tokenizer, "human_eval", max_tokens=100))
    
if __name__ == "__main__":
    main()

evaluators = {
    "mbpp": EvaluatorMBPP,
    "human_eval": EvaluatorHumanEval,
}