from abc import ABC, abstractmethod
from datasets import load_dataset

class LMDataset(ABC):
    dataset_name = None

    @abstractmethod
    def create_dataset(self, num_proc: int, seed: int, max_sample: int = None, do_split: bool = False, train_dataset: str = "train", test_dataset: str = "test") -> dict:
        dataset = load_dataset(self.dataset_name, num_proc=num_proc)
        if max_sample is not None:
            dataset["train"] = dataset[train_dataset].shuffle(seed=seed).select(range(max_sample))
            dataset["test"]  = dataset[test_dataset].shuffle(seed=seed).select(range(max_sample))
        if do_split:
            dataset = dataset[train_dataset].train_test_split(test_size=0.1, seed=seed)
        return {"dataset": dataset}

    
