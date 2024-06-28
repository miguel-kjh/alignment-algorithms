import os
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel

deberta_v3_large = 'microsoft/deberta-v3-large'
DATASET_FOLDER = "kaggle-llm-science-exam"

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def  main():
    name_main_train = "train.csv"
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, name_main_train))
    df_train = df_train.drop(columns="id")
    df_train = pd.concat([
        df_train,
        pd.read_csv(os.path.join(DATASET_FOLDER, "extra_train_set.csv")),
    ])
    #delete any rows with NaN or None values
    df_train = df_train.dropna()
    df_train = df_train.drop_duplicates()
    df_train.reset_index(inplace=True, drop=True)

    tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)
    option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    index_to_option = {v: k for k,v in option_to_index.items()}

    def preprocess(example):
        first_sentence = [example['prompt']] * 5
        second_sentences = [example[option] for option in 'ABCDE']
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
        tokenized_example['label'] = option_to_index[example['answer']]
        
        return tokenized_example

    dataset = Dataset.from_pandas(df_train)
    tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    #TRAIN
    training_args = TrainingArguments(
        warmup_ratio=0.8,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        report_to='none',
        output_dir='.',
    )

    model = AutoModelForMultipleChoice.from_pretrained(deberta_v3_large)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    #evaluation
    test_df = pd.read_csv('kaggle-llm-science-exam/test.csv')
    test_df['answer'] = 'A' # dummy answer that allows us to preprocess the test dataset just like we preprocessed the train dataset

    tokenized_test_dataset = Dataset.from_pandas(test_df.drop(columns=['id'])).map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E'])

    test_predictions = trainer.predict(tokenized_test_dataset).predictions
    test_predictions[:4]
    predictions_as_ids = np.argsort(-test_predictions, 1)
    predictions_as_ids[:3]
    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    predictions_as_answer_letters[:3]
    predictions_as_string = test_df['prediction'] = [
        ' '.join(row) for row in predictions_as_answer_letters[:, :3]
    ]
    predictions_as_string[:3]

    submission = test_df[['id', 'prediction']]
    submission.to_csv('submission.csv', index=False)
    pd.read_csv('submission.csv').head()


if __name__ == "__main__":
    main()