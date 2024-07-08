from abc import abstractmethod, ABC
from utils import RESPONSE_TEMPLATE

from datasets import load_dataset
from codebleu import calc_codebleu
from evaluate import load

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm

from utils import INTRUCTION_TEMPLATE

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

#delete warnings
import warnings
warnings.filterwarnings("ignore")

class Evaluator(ABC):
    
    def __init__(
            self,
            model,
            test_dataset,
            tokenizer,
        ) -> None:

        super().__init__()
        self._model: AutoModelForCausalLM = model
        self._tokenizer: AutoTokenizer = tokenizer
        self._test_dataset = test_dataset
        
    @staticmethod
    def generate_prompt(prompt):
        prompt = INTRUCTION_TEMPLATE + prompt
        answer = RESPONSE_TEMPLATE
        return prompt + answer
    
    @abstractmethod
    def evaluate(self, max_tokens: int, verbose: bool = True):
        pass
    

class EvaluatorMBPP(Evaluator):

    def evaluate(self, max_tokens: int, verbose: bool = True):
        try:
            self._model.to(device)
        except:
            pass
        self._model.eval()

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset
        code_eval = load("code_eval")
        rouge = load("rouge")
        meteor = load('meteor')
        pass_at_k_all = []
        y_hat = []
        y = []

        for example in iterator:
            prompt = self.generate_prompt(example["prompt"])
            real_code = example["code"]

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = output_text.split(RESPONSE_TEMPLATE)[1]
            y_hat.append(code)
            y.append(real_code)
            """rouge_results = rouge.compute(predictions=code, references=real_code)
            print(rouge_results)
            exit()"""
            #candidates = [[code]]
            #test_cases = [example["test_list"][0]]
            #print(prompt)
            #print(candidates[0][0])
            #print("#"*10, "Real code and test cases", "#"*10)
            #print(real_code)
            #print(test_cases)
            #exit()
            #pass_at_k, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            #pass_at_k_all.append(pass_at_k["pass@1"])
        
        results = calc_codebleu(y, y_hat, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        rouge_results = rouge.compute(predictions=y_hat, references=y)
        meteor_results = meteor.compute(predictions=y_hat, references=y)
        metrics = {
            "codebleu": results['codebleu'],
            "rouge1": rouge_results['rouge1'],
            "meteor": meteor_results['meteor'],
            #"pass@1": sum(pass_at_k_all) / len(pass_at_k_all),
        }

        return metrics

evaluators = {
    "mbpp": EvaluatorMBPP,
}

def evaluate_model(model, dataset, tokenizer, name_of_evluator, max_tokens=100) -> dict:
    evaluator = evaluators[name_of_evluator](model, dataset, tokenizer)
    metrics = evaluator.evaluate(max_tokens=max_tokens)
    return {
        name: round(result*100, 2)
        for name, result in metrics.items()
    }
    
def main():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", num_proc=10, split="test")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = AutoModelForCausalLM.from_pretrained("saved_models/code_model/pythia-1b-deduped_tuning_code_epoch_5_lr_0.0001_wd_0.01_bs_4_block_256_timestamp_2024-07-07_00-59-15/checkpoint-22525")
    print(evaluate_model(model, ds, tokenizer, "mbpp", max_tokens=100))
    
if __name__ == "__main__":
    main()