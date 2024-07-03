from utils import RESPONSE_TEMPLATE

from datasets import load_dataset
from evaluate import load

from transformers import GPTNeoXForCausalLM, AutoTokenizer
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

class Evaluator:

    def __init__(
            self,
            model,
            test_dataset,
            tokenizer,
        ) -> None:

        super().__init__()
        self._model: GPTNeoXForCausalLM = model
        self._tokenizer: AutoTokenizer = tokenizer
        self._test_dataset = test_dataset

    @staticmethod
    def generate_prompt(prompt):
        prompt = INTRUCTION_TEMPLATE + prompt
        answer = RESPONSE_TEMPLATE
        return prompt + answer

    def evaluate(self, max_tokens: int, verbose: bool = True):
        self._model.to(device)

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset
        code_eval = load("code_eval")
        pass_at_k_all = []

        for example in iterator:
            prompt = self.generate_prompt(example["prompt"])

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
            candidates = [[code]]
            test_cases = [example["test_list"][0]]
            pass_at_k, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            pass_at_k_all.append(pass_at_k["pass@1"])

        metrics = {
            "pass@1": sum(pass_at_k_all) / len(pass_at_k_all),
        }

        return metrics


def evaluate_model(model, dataset, tokenizer, max_tokens=100) -> dict:
    evaluator = Evaluator(model, dataset, tokenizer)
    metrics = evaluator.evaluate(max_tokens=max_tokens)
    return {
        name: round(result, 2)*100
        for name, result in metrics.items()
    }
    
def main():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", num_proc=10, split="test")
    model = GPTNeoXForCausalLM.from_pretrained("saved_models/code_model/pythia-70m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    print(evaluate_model(model, ds, tokenizer, max_tokens=50))
    
if __name__ == "__main__":
    main()