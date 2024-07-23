from transformers import AutoModelForCausalLM, AutoTokenizer

from EvaluatorHumanEval import EvaluatorHumanEval
from EvaluatorMBPP import EvaluatorMBPP
from EvaluatorCommonsenQA import EvaluatorCommonsenQA

import os

from utils import calculate_metrics
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#delete warnings
import warnings
warnings.filterwarnings("ignore")

evaluators = {
    "mbpp": EvaluatorMBPP,
    "human_eval": EvaluatorHumanEval,
    "commonsense_qa": EvaluatorCommonsenQA,
}

def evaluate_model(model, tokenizer, name_of_evluator, max_tokens=100) -> dict:
    evaluator = evaluators[name_of_evluator](model, tokenizer)
    results = evaluator.evaluate(max_tokens=max_tokens)
    if name_of_evluator in ["mbpp", "human_eval"]:
        metrics = calculate_metrics(*results)
        return {
            name: round(result*100, 2)
            for name, result in metrics.items()
        }
    else:
        return results

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = AutoModelForCausalLM.from_pretrained("saved_models/code_model/pythia-410m-deduped_tuning_code_epoch_5_lr_0.0001_wd_0.01_bs_8_block_512_timestamp_2024-07-10_21-53-28_idda_lima/checkpoint-11265")
    print(evaluate_model(model, tokenizer, "human_eval", max_tokens=100))
    
if __name__ == "__main__":
    main()