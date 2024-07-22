from LMDataset import LMDataset
from utils import generate_sample


class CommonsenseQA(LMDataset):

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "tau/commonsense_qa"

    def create_dataset(self, num_proc: int, seed: int, max_sample: int = None, do_split: bool = False) -> dict:
        dataset_dict = super().create_dataset(num_proc, seed, max_sample, do_split, test_dataset="validation")

        def format_prompt_completions(example):
            formatted_texts = []
            for question,data,answerKey in zip(example["question"], example["choices"], example["answerKey"]):
                formatted_string = ""
                for label, text in zip(data['label'], data['text']):
                    formatted_string += f"({label.lower()}) {text} "
                formatted_string = question + formatted_string.strip()
                completion = f"{answerKey.lower()}"
                formatted_texts.append(generate_sample(formatted_string, completion))
            return formatted_texts

        dataset_dict["format_prompt_completions"] = format_prompt_completions
        return dataset_dict