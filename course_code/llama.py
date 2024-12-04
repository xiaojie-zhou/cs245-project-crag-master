import json
import numpy as np
import torch
from transformers import pipeline
from bs4 import BeautifulSoup



def summarize_prompt(text):
    messages = [
        {"role": "system", "content": "You are summarizing the provided text."},
        {"role": "user", "content": f"Please summarize the following text:\n{text}"}
    ]
    return messages

def classify_question_topic(text):
    messages = [
        {"role": "system", "content": "Please select the topic to which the provided question is related."},
        {"role": "user", "content": f"Please choose a topic for the following question:\n{text}"},
    ]
    return messages

def classify_prompt(text):
    messages = [
        {"role": "system", "content": "You are summarizing the provided text and decides which topic it falls into."},
        {"role": "user", "content": f"Please choose a topic for the following text:\n{text}"},
    ]
    return messages
"""
file_path = "/data_weizhen/cs245-project-crag-master/data/crag_task_1_dev_v4_release.jsonl"
data = []
with open(file_path, "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))
        break
question = data[-1]["query"]
html_source = data[-1]["search_results"][0]["page_result"]
soup = BeautifulSoup(html_source, "lxml")
for tag in soup(["script", "style", "meta", "link"]):
    tag.decompose()
text = soup.get_text(separator=" ", strip=True)
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=0
)

outputs = pipe(
    summarize_prompt(text),
    max_new_tokens=256,
)
summarization = outputs[0]["generated_text"][-1]


outputs = pipe(
    classify_prompt(summarization),
    max_new_tokens=256,
)
topic = outputs[0]["generated_text"][-1]

print(topic)
outputs = pipe(
    [classify_question_topic(x) for x in [question, question]],
    max_new_tokens=256,
)
question_topic = outputs#outputs[0]["generated_text"][-1]

print(len(outputs))
print(question_topic)"""
