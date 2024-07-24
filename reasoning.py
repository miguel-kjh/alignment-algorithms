from datasets import load_dataset
from token_id import OPEN_IA_API_KEY
from langchain_openai import ChatOpenAI
import os
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tqdm


os.environ["OPENAI_API_KEY"] = OPEN_IA_API_KEY
MODEL = "gpt-4o-mini"
NUMBER_OF_SAMPLES = 100

model = ChatOpenAI(model=MODEL)
test_dataset = load_dataset("commonsense_qa", split="train")
#shuffle the dataset
test_dataset = test_dataset.shuffle()
test_dataset = test_dataset.select(range(NUMBER_OF_SAMPLES))


template_prompt = """
Given a multiple-choice question along with the options and the correct answer, provide me with the reasoning behind the correct answer.
Format:
Question: {question}
Options: {options}
Correct Answer: {correct_answer}
Reasoning:
"""

prompt = ChatPromptTemplate.from_template(template_prompt)
chain = prompt | model | StrOutputParser()

data = []
for example in tqdm.tqdm(test_dataset, desc="Generating reasoning"):
    options = ""
    for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
        options += f"({label.lower()}) {text} "
    input_data = {
        "question": example["question"],
        "options": options.strip(),
        "correct_answer": example["answerKey"].lower()
    }
    reasoning = chain.invoke(input_data).strip().split("Reasoning:")[-1]
    # Agregar los datos a la lista
    data.append({
        "question": example["question"] + options.strip(),
        "answer": f"{reasoning}. The correct answer is {example['answerKey']}"
    })

df = pd.DataFrame(data)
df.to_excel("questions_with_reasoning.xlsx", index=False)

print("Excel file has been created successfully.")

