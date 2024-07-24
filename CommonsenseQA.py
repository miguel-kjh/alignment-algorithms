import pandas as pd
from LMDataset import LMDataset
from utils import generate_sample
from datasets import Dataset


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
    
class CommonsenseQARationale(LMDataset):

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "questions_with_reasoning.xlsx"
        self.df = pd.read_excel(self.dataset_name)
        
    def add_data(self, data: dict):
        self.df = self.df.append(data, ignore_index=True)

    def create_dataset(self) -> dict:
        rationale_dataset = Dataset.from_pandas(self.df)
        dataset_dict = {"dataset": rationale_dataset}

        def format_prompt_completions(example):
            formatted_texts = []
            for question,data,answerKey,rationale in zip(example["question"], example["options"], example["correct_answer"], example["reasoning"]):
                formatted_string = question + data
                completion = f"{rationale} The correct answer is {answerKey}"
                formatted_texts.append(generate_sample(formatted_string, completion))
            return formatted_texts

        dataset_dict["format_prompt_completions"] = format_prompt_completions
        return dataset_dict
    
class CommonsenseQAFewShot(LMDataset):

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "tau/commonsense_qa"
        reasoning_data = "questions_with_reasoning.xlsx"
        df = pd.read_excel(reasoning_data)
        self.rationale_dataset = Dataset.from_pandas(df)
        
    def _initialize_prompt(self):
        template_prompt = ""
        for example in self.rationale_dataset:
            template_prompt += f"Question: {example['question']}\n Options: {example['options']}\n Reasoning: {example['reasoning']}\n The correct answer is: {example['correct_answer']}\n"
        return template_prompt

    def create_dataset(self, num_proc: int, seed: int, max_sample: int = None, do_split: bool = False) -> dict:
        dataset_dict = super().create_dataset(num_proc, seed, max_sample, do_split, test_dataset="validation")

        def format_prompt_completions(example):
            formatted_texts = []
            for question,data,answerKey in zip(example["question"], example["choices"], example["answerKey"]):
                formatted_string = self._initialize_prompt()
                for label, text in zip(data['label'], data['text']):
                    formatted_string += f"({label.lower()}) {text} "
                formatted_string = question + formatted_string.strip()
                formatted_texts.append((generate_sample(formatted_string, ""), answerKey.lower()))
            return formatted_texts

        dataset_dict["format_prompt_completions"] = format_prompt_completions
        return dataset_dict
    
if __name__ == "__main__":
    rationale_dataset = CommonsenseQAFewShot()
    ds = rationale_dataset.create_dataset(10, 20)
    print(ds)
    example = ds["dataset"]['train']
    print(example)
    f = ds["format_prompt_completions"]
    print(f(example)[0])