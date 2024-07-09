from Evaluator import Evaluator


import torch
import tqdm
from datasets import load_dataset


class EvaluatorHumanEval(Evaluator):

    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        self._test_dataset = load_dataset(
            "openai/openai_humaneval",
            num_proc=10,
        )["test"]

    def evaluate(self, max_tokens: int, verbose: bool = True):
        try:
            self._model.to(self._device)
        except:
            pass
        self._model.eval()

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset

        y_hat = []
        y = []
        tests = []

        for example in iterator:
            prompt = self.generate_prompt(example["prompt"])
            real_code = example["prompt"] + example["canonical_solution"]

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            code = output_text.replace("### Human:", "").replace("### Response:", "")
            y_hat.append(code)
            y.append(real_code)
            tests.append(example["test"])

        return y_hat, y, tests