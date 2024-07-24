import torch
import tqdm
from Evaluator import Evaluator
from utils import INTRUCTION_TEMPLATE, RESPONSE_TEMPLATE


from datasets import load_dataset
import evaluate


class EvaluatorCommonsenQA(Evaluator):

    def __init__(self, model, tokenizer, using_start) -> None:
        super().__init__(model, tokenizer)
        self._test_dataset = load_dataset(
            "commonsense_qa",
            num_proc=10,
        )["validation"]
        self._using_start = using_start

    def evaluate(self, max_tokens: int, verbose: bool = True):
        try:
            self._model.to(self._device)
        except:
            pass
        self._model.eval()

        iterator = tqdm.tqdm(self._test_dataset, desc="Evaluating") if verbose else self._test_dataset

        y_hat = []
        y = []

        for example in iterator:
            prompt = ""
            for label, text in zip(example["choices"]['label'], example["choices"]['text']):
                prompt += f"({label.lower()}) {text} "
            prompt = example["question"] + prompt.strip()
            prompt = self.generate_prompt(prompt)
            real_answer = example["answerKey"].lower()

            inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = output_text.split(RESPONSE_TEMPLATE)[1].strip().lower()
                if self._using_start:
                    #get the answer from the start of the output using `The correct answer is`
                    answer = output_text.split("The correct answer is")[-1].strip().lower()

            y_hat.append(answer)
            y.append(real_answer)

        # Calculate accuracy
        correct = 0
        for pred, real in zip(y_hat, y):
            if pred == real:
                correct += 1
        return {"accuracy": correct * 100 / len(y)}