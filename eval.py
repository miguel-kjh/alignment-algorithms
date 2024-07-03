from utils import RESPONSE_TEMPLATE

from datasets import load_dataset
from evaluate import load

import torch
import tqdm

from utils import INTRUCTION_TEMPLATE

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class Evaluator:

    def __init__(
            self,
            model,
            test_dataset,
            tokenizer,
        ) -> None:

        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._test_dataset = test_dataset

    @staticmethod
    def generate_prompt(prompt):
        prompt = INTRUCTION_TEMPLATE + prompt
        answer = RESPONSE_TEMPLATE
        return prompt + answer

    def evaluate(self, max_tokens: int, verbose: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset

        y_true = []
        y_pred = []

        for example in iterator:
            y_true.append(example["canonical_solution"])
            prompt = self.generate_prompt(example["prompt"])

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        metrics = {}

        return metrics


def evaluate_model(model, dataset, tokenizer, max_tokens=1) -> dict:
    evaluator = Evaluator(model, dataset["test"], tokenizer)
    metrics = evaluator.evaluate(max_tokens=max_tokens)
    return {
        name: round(result, 2)*100
        for name, result in metrics.items()
    }
    
if __name__ == "__main__":

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", num_proc=10, split=["train", "test", "validation"])
    train_dataset = ds[0]
    idx = 0
    print("train_dataset")
    print(train_dataset["prompt"][idx])
    print("code")
    print(train_dataset["code"][idx])
    print("test_list")
    print(train_dataset["test_list"][idx])
    code_eval = load("code_eval")
    candidates = [[train_dataset["code"][idx+1]]]
    test_cases = [train_dataset["test_list"][idx][0]]
    pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
    print(pass_at_k)