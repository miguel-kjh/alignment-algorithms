import torch
from utils import INTRUCTION_TEMPLATE, RESPONSE_TEMPLATE


from transformers import AutoModelForCausalLM, AutoTokenizer


from abc import ABC, abstractmethod


class Evaluator(ABC):

    def __init__(
            self,
            model,
            tokenizer,
        ) -> None:

        super().__init__()
        self._model: AutoModelForCausalLM = model
        self._tokenizer: AutoTokenizer = tokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    @staticmethod
    def generate_prompt(prompt):
        prompt = INTRUCTION_TEMPLATE + prompt
        answer = RESPONSE_TEMPLATE
        return prompt + answer

    @abstractmethod
    def evaluate(self, max_tokens: int, verbose: bool = True):
        pass