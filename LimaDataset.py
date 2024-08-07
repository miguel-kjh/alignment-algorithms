from LMDataset import LMDataset
from utils import generate_sample


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