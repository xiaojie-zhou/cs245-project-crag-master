import json
import numpy as np
from transformers import pipeline
def cosine_sim(arr1, arr2):
    return np.dot(arr1,arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2))


file_path = "/data_weizhen/cs245-project-crag-master/data/crag_task_1_dev_v4_release.jsonl"
data = []
candidate_topics = ["finance", "music", "movie", "sports", "others"]
with open(file_path, "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))
        break

# Now `data` is a list of dictionaries
#print(data[-1].keys())
#print(data[-1]["domain"])
#print(data[-1]["query"])
#print(data[-1]["search_results"][0])
#print(data[-1]["answer"])

question = data[-1]["query"]

html_source = data[-1]["search_results"][0]["page_result"]
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_source, "lxml")
# Remove unnecessary tags
for tag in soup(["script", "style", "meta", "link"]):
    tag.decompose()
# Extract visible text
text = soup.get_text(separator=" ", strip=True)
from pprint import pprint
#print(text)

def summarize(text):
    messages = [
        {"role": "system", "content": "You are summarizing the provided text."},
        {"role": "user", "content": f"Please summarize the following text:\n{text}"},
    ]
    return messages


# Initialize a summarization pipeline
summarizer = pipeline("summarization", model="meta-llama/Llama-3.2-3B", device=0)
classifier = pipeline("zero-shot-classification", model="meta-llama/Llama-3.2-3B", device=1)
# Summarize text (split into chunks if necessary to meet model limits)
max_chunk_size = 1024
chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
summary = []
for chunk in chunks:
    summarized_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
    summary.append(summarized_chunk[0]['summary_text'])

# Combine the summaries into paragraphs
final_summary = "\n\n".join(summary)


results = classifier(final_summary, candidate_topics)
print(results)
probabilities = {label: score for label, score in zip(results['labels'], results['scores'])}
ordered_probabilities_1 = np.array([probabilities[topic] for topic in candidate_topics])
print(ordered_probabilities_1)

# Save to a file
output_file = "summarized_paragraphs.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(final_summary)


results = classifier(question, candidate_topics)
print(results)
probabilities = {label: score for label, score in zip(results['labels'], results['scores'])}
ordered_probabilities_2 = np.array([probabilities[topic] for topic in candidate_topics])
print(ordered_probabilities_2)



print(cosine_sim(ordered_probabilities_1, ordered_probabilities_2))


# First, decides the topic of a search results.
    #Summzirize the html search files into a paragrah
    #Decides the topic of the summarization
    #Encode the topic v1
# Second
    #Decides the topic of the query
    #Encode the topic v2

# Multiply the chunk-level cosine similarity by cos(v1, v2)

