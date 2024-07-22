from abc import ABC, abstractmethod
from datasets import load_dataset
from utils import generate_sample

class LMDataset(ABC):
    dataset_name = None

    @abstractmethod
    def create_dataset(self, num_proc: int, seed: int, max_sample: int = None, do_split: bool = False, train_dataset: str = "train", test_dataset: str = "test") -> dict:
        dataset = load_dataset(self.dataset_name, num_proc=num_proc)
        if max_sample is not None:
            dataset[train_dataset] = dataset[train_dataset].shuffle(seed=seed).select(range(max_sample))
            dataset[test_dataset]  = dataset[test_dataset].shuffle(seed=seed).select(range(max_sample))
        if do_split:
            dataset = dataset[train_dataset].train_test_split(test_size=0.1, seed=seed)
        return {"dataset": dataset}

class CodeAlpacaDataset(LMDataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "HuggingFaceH4/CodeAlpaca_20K"

    def create_dataset(self, num_proc: int, seed: int, max_sample: int = 100, do_split: bool = False) -> dict:
        dataset_dict = super().create_dataset(num_proc, seed, max_sample, do_split)
        
        def format_prompt_completions(example):
            formatted_texts = []
            for prompt, completion in zip(example["prompt"], example["completion"]):
                formatted_texts.append(generate_sample(prompt, completion))
            return formatted_texts
        
        dataset_dict["format_prompt_completions"] = format_prompt_completions        
        return dataset_dict
    
class LimaDataset(LMDataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "GAIR/lima"

    def create_dataset(self, num_proc: int, seed: int, max_sample: int = 100, do_split: bool = False) -> dict:
        dataset_dict = super().create_dataset(num_proc, seed, max_sample, do_split)
        
        def format_prompt_completions(example):
            formatted_texts = []
            for sample in zip(example["conversations"]):
                prompt = sample[0][0]
                completion = sample[0][1]
                formatted_texts.append(generate_sample(prompt, completion))
            return formatted_texts
        
        dataset_dict["format_prompt_completions"] = format_prompt_completions        
        return dataset_dict
    
class CommonsenseQA(LMDataset):

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "tau/commonsense_qa"

#main
if __name__ == "__main__":
    ds = load_dataset("tau/commonsense_qa", num_proc=10)
    print(ds)
